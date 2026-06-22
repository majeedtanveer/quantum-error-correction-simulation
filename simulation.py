import stim, pymatching
from abc import ABC, abstractmethod
import sinter
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from typing import List, Iterable


def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    """
    Estimate the number of logical errors produced by a quantum circuit.

    The circuit is sampled `num_shots` times. Detection events are decoded
    using a minimum-weight perfect matching decoder (PyMatching), and the
    predicted logical observables are compared to the true outcomes.

    Parameters
    ----------
    circuit : stim.Circuit
        The quantum circuit to simulate.
    num_shots : int
        Number of independent samples (shots) to run.

    Returns
    -------
    int
        The number of shots for which the decoded logical observable
        differs from the actual observable.
    """
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    predictions = matcher.decode_batch(detection_events)

    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1

    return num_errors

def repetition_theory(p: float, d: int) -> float:
    """
    Compute the theoretical logical error rate of a repetition code.

    The logical failure probability is given by the probability that
    strictly more than half of the `d` physical qubits experience
    a bit-flip error.

    Parameters
    ----------
    p : float
        Physical bit-flip error probability per qubit.
    d : int
        Code distance (number of physical qubits).

    Returns
    -------
    float
        Logical error probability.

    Notes
    -----
    This is equivalent to the upper tail of a Binomial(d, n) distribution:
        P(logical error) = sum_{k = (n+1)//2}^{n} C(n, j) p^j (1-p)^(n-j)
    """
    return sum(
        comb(d, j) * p**j * (1 - p)**(d - j)
        for j in range((d + 1) // 2, d + 1)
    )

class CircuitBuilder(ABC):
    """
    Abstract base class for quantum error-correction circuit construction.

    Subclasses must implement the `build_circuit` method.
    """

    @abstractmethod
    def build_circuit(self, distance: int, noise: dict) -> stim.Circuit:
        """
        Construct a quantum circuit.

        Parameters
        ----------
        distance : int
            Code distance parameter.
        noise : dict
            Dictionary specifying noise parameters.

        Returns
        -------
        stim.Circuit
            The constructed quantum circuit.
        """
        pass


#   TODO:
#     - concantenation option (Shor's Code)
#       - build with repetitioncodebuilder

class RepetitionCodeBuilder(CircuitBuilder):
    """
    Builder for repetition code circuits.

    Constructs a `[d,1,d]` distance-`d` repetition code with bit-flip noise,
    measurement, detector definitions, and a logical observable. 
    The distance is related to the logical X_L Error.

    TODO: 
    - phase flip option

    """

    def build_circuit(self, distance, noise, logical_one=False):
        """
        Build a repetition code circuit.

        Parameters
        ----------
        distance : int
            Code distance. Must be an odd integer ≥ 3.
        noise : dict
            Noise specification. Expected key:
            - "x": float
                Probability of bit-flip (X) error per qubit.
        logical_one : bool, optional
            If True, prepares the logical |1⟩ state instead of |0⟩.

        Returns
        -------
        stim.Circuit
            The constructed repetition code circuit.

        Raises
        ------
        ValueError
            If the distance is not a valid odd integer ≥ 3.
        """
        if distance % 2 == 0 or distance < 3 or not isinstance(distance, int):
            raise ValueError("Invalid Distance!")

        circuit = stim.Circuit()

        circuit.append("R", list(range(distance)))

        if logical_one:
            circuit.append("X", 0)

        for d in range(1, distance):
            circuit.append("CNOT", [0, d])

        circuit.append("X_ERROR", list(range(distance)), noise["x"])

        circuit.append("M", list(range(distance)))
        for d in range(1, distance):
            circuit.append(
                "DETECTOR",
                [stim.target_rec(-1 * (d + 1)), stim.target_rec(-1 * d)],
            )

        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

        return circuit


class SinisterSimulation:
    """
    Driver class for running quantum error-correction simulations.

    This class generates parameter sweeps over code distances and
    physical error rates, executes simulations using Sinter, and
    provides visualization utilities.
    """

    def __init__(
        self,
        circuit_builder,
        distances: Iterable[int] = (3, 5, 7, 9),
        noise_values: Iterable[float] = (0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5),
        num_workers: int = 4,
        max_shots: int = 100_000,
        max_errors: int = 500,
    ):
        """
        Initialize simulation parameters.

        Parameters
        ----------
        circuit_builder : CircuitBuilder
            Instance responsible for constructing circuits.
        distances : Iterable[int], optional
            Sequence of code distances to evaluate.
        noise_values : Iterable[float], optional
            Sequence of physical error probabilities.
        num_workers : int, optional
            Number of parallel workers used by Sinter.
        max_shots : int, optional
            Maximum number of samples per task.
        max_errors : int, optional
            Maximum number of logical errors per task before stopping.
        """
        self.circuit_builder = circuit_builder
        self.distances = distances
        self.noise_values = noise_values
        self.num_workers = num_workers
        self.max_shots = max_shots
        self.max_errors = max_errors

    def run_single(self, d: int, p: float) -> float:
        """
        Execute a single logical error rate simulation.

        A circuit is constructed for the specified code distance and
        physical error rate, simulated for `max_shots` samples, and
        decoded using PyMatching. The logical error rate (LER) is
        computed as the fraction of shots resulting in a logical failure.

        Parameters
        ----------
        d : int
            Code distance.
        p : float
            Physical error probability.

        Returns
        -------
        float
            Estimated logical error rate per shot.
        """
        num_errors = count_logical_errors(
            circuit=self.circuit_builder.build_circuit(d, {"x": p}),
            num_shots=self.max_shots,
        )

        return num_errors / self.max_shots


    def run(self) -> np.ndarray:
        """
        Execute simulations for all distance and noise combinations.

        Logical error rates are estimated for every pair of code
        distance and physical error probability defined by
        `self.distances` and `self.noise_values`.

        Returns
        -------
        np.ndarray
            Two-dimensional array of logical error rates with shape

                (len(self.distances), len(self.noise_values))

            where rows correspond to code distances and columns
            correspond to physical error probabilities.
        """
        return np.array([
            [self.run_single(d, p) for p in self.noise_values]
            for d in self.distances
        ])




    def plot(self, ler, theory_vals=None, title="QEC-Code"):
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, d in enumerate(self.distances):
            ax.plot(
                self.noise_values,
                ler[i],
                marker="o",
                linewidth=2,
                label=f"d={d}",
            )

        if theory_vals is not None:
            ps = np.geomspace(
                min(self.noise_values),
                max(self.noise_values),
                300
            )

            for d in self.distances:
                ax.plot(
                    ps,
                    [theory_vals(p, d) for p in ps],
                    "--",
                    alpha=0.7,
                    label=f"d={d} theory",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim(
            min(self.noise_values),
            max(self.noise_values),
        )

        ax.set_ylim(
            max(np.min(ler[ler > 0]) / 2, 1e-6),
            1,
        )

        ax.set_title(f"{title} Error Rates")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate per Shot")

        ax.grid(which="major")
        ax.grid(which="minor", alpha=0.3)

        ax.legend()
        fig.set_dpi(120)

        plt.tight_layout()
        plt.show()