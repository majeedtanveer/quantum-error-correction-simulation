import collections
import stim, pymatching
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from typing import List, Iterable
from scipy.special import comb as scipy_comb

from StimCircuits.stimcircuits.surface_code import generate_circuit

@dataclass(frozen=True)
class CodeCircuit:
    circuit: stim.Circuit
    name: str
    # can be either X or Z depending on the type of memory experiment
    memory_type: str
    n: int
    k: int
    d: int

@dataclass(frozen=True)
class LogicalErrorSample:
    num_errors: int
    predictions: np.ndarray
    detection_events: np.ndarray
    observable_flips: np.ndarray


@dataclass(frozen=True)
class SyndromeResamplingResult:
    standard_logical_error_rate: float
    resampled_logical_error_rate: float
    logical_error_mask: np.ndarray
    syndrome_probabilities: np.ndarray
    weights: np.ndarray

def get_syndrome_probabilities(syndromes: np.ndarray, alpha: int) -> np.ndarray:
    """
    Compute the probability of each syndrome in a set of detection events.

    Parameters
    ----------
    syndromes : np.ndarray
        Array of shape (num_shots, num_detectors) representing detection events.

    Returns
    -------
    np.ndarray
        Array of shape (num_shots,) containing the probability of each syndrome.
    """
    values = np.asarray(syndromes)
    if values.ndim == 2:
        values = values[:, np.newaxis, :]
    elif values.ndim != 3:
        raise ValueError(
            "syndromes must have shape (shots, detectors) or "
            "(shots, rounds, detectors)."
        )

    if not isinstance(alpha, int) or alpha < 1:
        raise ValueError("alpha must be a positive integer.")

    _, _, num_detectors = values.shape
    counts = np.zeros(2**num_detectors, dtype=int)

    for shot in count_syndromes_per_shot(values):
        for syndrome, count in shot.items():
            # print("syndrome:", syndrome, "count:", count)
            counts[int(syndrome, 2)] += count

    total_samples = counts.sum()
    if total_samples < alpha:
        raise ValueError("There must be at least alpha syndrome samples.")

    power_distribution = scipy_comb(counts, alpha, exact=False)
    normalization = scipy_comb(total_samples, alpha, exact=False)

    return power_distribution / normalization


def plot_q_alpha_distribution(
    syndromes: np.ndarray,
    alphas: Iterable[int] = (1, 2, 3),
    title: str = "Syndrome power distribution",
) -> None:
    """Plot sorted Q_alpha(s) values from one syndrome per shot."""
    values = np.asarray(syndromes)

    fig, ax = plt.subplots(figsize=(7, 5))
    for alpha in alphas:
        probabilities = get_syndrome_probabilities(values, int(alpha))
        distribution = probabilities / np.sum(probabilities)
        sorted_distribution = np.sort(distribution[distribution > 0])[::-1]
        ax.plot(
            np.arange(1, len(sorted_distribution) + 1),
            sorted_distribution,
            ".-",
            label=fr"$\alpha={alpha}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Syndrome index (sorted by probability)")
    ax.set_ylabel(r"Probability $Q_\alpha(s)$")
    ax.set_title(title)
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def encode_syndromes(syndromes: np.ndarray) -> list[list[str]]:
    """Encode each shot as a list of fixed-width binary strings, one per round."""
    values = np.asarray(syndromes)
    if values.ndim == 0:
        values = values.reshape(1, 1, 1)
    elif values.ndim == 1:
        values = values.reshape(1, 1, -1)
    else:
        values = values.reshape(values.shape[0], values.shape[1], -1)

    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Syndromes must contain only 0 and 1 values.")

    return [
        ["".join(str(int(bit)) for bit in round_values) for round_values in shot]
        for shot in values
    ]


def count_syndromes_per_shot(
    syndromes: np.ndarray,
) -> list[dict[str, int]]:
    """Count each round-level syndrome separately within every shot."""
    encoded_syndromes = encode_syndromes(syndromes)
    return [dict(collections.Counter(shot)) for shot in encoded_syndromes]


def syndrome_resample_logical_errors(
    circuit: stim.Circuit | CodeCircuit,
    num_shots: int,
    alpha: float = 2.0,
    code: CodeCircuit | None = None,
) -> SyndromeResamplingResult:
    """
    Estimate a syndrome-resampled logical error rate.

    This follows the standard importance-weighting pattern used in rare-event
    syndrome sampling: estimate the syndrome probability distribution P(s), then
    reweight each shot by P(s)^(alpha - 1) before averaging the decoder error
    indicator.

    Parameters
    ----------
    circuit : stim.Circuit
        The quantum circuit to sample.
    num_shots : int
        Number of independent Monte Carlo samples.
    alpha : float, optional
        Resampling exponent. ``alpha=1`` gives the usual logical error rate.

    Returns
    -------
    SyndromeResamplingResult
        Contains the standard and resampled logical error rate estimates, the
        per-shot logical error mask, the syndrome probabilities, and the
        per-shot importance weights.
    """
    if isinstance(circuit, CodeCircuit):
        code = circuit
        circuit = code.circuit

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    syndromes = get_syndromes_from_detection_events(code, detection_events)

    probabilities = get_syndrome_probabilities(syndromes, int(alpha))
    weights = probabilities ** (alpha - 1.0)

    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    predictions = matcher.decode_batch(detection_events)

    logical_error_mask = np.any(predictions != observable_flips, axis=1)

    resampled_rate = float(
        np.sum(weights * logical_error_mask.astype(float)) / np.sum(weights)
    )
    # else:
    #     detector_coordinates = get_circuit_coordinates(circuit)["detectors"]
    #     if detector_coordinates and len(detector_coordinates[0]["coordinates"]) >= 3:
    #         final_round = max(
    #             detector["coordinates"][2]
    #             for detector in detector_coordinates
    #         )
    #         final_detector_indices = [
    #             index
    #             for index, detector in enumerate(detector_coordinates)
    #             if detector["coordinates"][2] == final_round
    #         ]
    #         final_round_syndromes = detection_events[:, final_detector_indices]
    #     else:
    #         final_round_syndromes = detection_events
    #     encoded_syndromes = encode_syndromes(final_round_syndromes)
    #     syndrome_indices = np.array(
    #         [int(shot[0], 2) for shot in encoded_syndromes],
    #         dtype=int,
    #     )
    #     q_distribution = get_syndrome_probabilities(
    #         final_round_syndromes, int(alpha)
    #     )
    #     counts = np.bincount(
    #         syndrome_indices,
    #         minlength=q_distribution.size,
    #     )
    #     empirical_distribution = counts / num_shots
    #     syndrome_probabilities = empirical_distribution[syndrome_indices]
    #     weights = np.divide(
    #         q_distribution[syndrome_indices],
    #         syndrome_probabilities,
    #         out=np.zeros(num_shots, dtype=float),
    #         where=syndrome_probabilities > 0,
    #     )

    # detector_error_model = circuit.detector_error_model(decompose_errors=True)
    # matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    # predictions = matcher.decode_batch(detection_events)

    # logical_error_mask = np.any(predictions != observable_flips, axis=1)
    # if weights.sum() == 0:
    #     resampled_rate = 0.0
    # else:
    #     resampled_rate = float(
    #         np.sum(weights * logical_error_mask.astype(float)) / np.sum(weights)
    #     )

    # return SyndromeResamplingResult(
    #     standard_logical_error_rate=float(np.mean(logical_error_mask)),
    #     resampled_logical_error_rate=resampled_rate,
    #     logical_error_mask=logical_error_mask.astype(bool),
    #     syndrome_probabilities=syndrome_probabilities,
    #     weights=weights,
    # )


def sample_logical_errors(
    circuit: stim.Circuit,
    num_shots: int,
) -> LogicalErrorSample:
    """
    Sample and decode a quantum circuit.

    The circuit is sampled `num_shots` times. Detection events are decoded
    using a minimum-weight perfect matching decoder (PyMatching), and the
    predicted logical observables are compared to the true outcomes.

    Parameters
    ----------
    circuit : stim.Circuit
        The quantum circuit to simulate.
    num_shots : int
        Number of independent samples (shots) to run.

    Returns a structured result containing the raw detector samples,
    observable flips, decoder predictions, optional decoder weights, and
    the total logical error count.
    """
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    predictions = matcher.decode_batch(
        detection_events
    )

    num_errors = 0

    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1

    return LogicalErrorSample(
        num_errors=num_errors,
        predictions=predictions,
        detection_events=detection_events,
        observable_flips=observable_flips
    )

def get_circuit_coordinates(circuit: stim.Circuit) -> dict[str, list[dict]]:
    """Return qubit and detector coordinates recorded in a Stim circuit."""
    qubits = []
    detectors = []

    for instruction_index, instruction in enumerate(circuit.flattened()):
        if instruction.name == "QUBIT_COORDS":
            for target in instruction.targets_copy():
                qubits.append({
                    "instruction_index": instruction_index,
                    "qubit": target.value,
                    "coordinates": instruction.gate_args_copy(),
                })
        elif instruction.name == "DETECTOR":
            detectors.append({
                "instruction_index": instruction_index,
                "coordinates": instruction.gate_args_copy(),
                "targets": instruction.targets_copy(),
            })

    return {
        "qubits": qubits,
        "detectors": detectors,
    }


def get_syndromes_from_detection_events(code: CodeCircuit, detection_events: np.ndarray) -> np.ndarray:
    """
    Convert detection events to syndrome representation. 
    This works for surface code P-type memory experiments (P is either X or Z).

    Parameters
    ----------
    detection_events : np.ndarray
        Array of detection events with shape (num_shots, num_detectors).
    num_syndromes_per_round : int
        Number of syndromes per round.

    Returns
    -------
    np.ndarray
        Array of syndromes with shape (num_shots, num_detectors).
        Each entry is 1 if the corresponding detector fired, 0 otherwise.
    """

    circuit = code.circuit
    instructions = get_circuit_coordinates(circuit)
    detectors = instructions["detectors"]
    qubits = instructions["qubits"]

    qubit_by_xy = {
        tuple(qubit["coordinates"][:2]): qubit["qubit"]
        for qubit in qubits
    }

    memory_type = code.memory_type
    other_type = "Z" if memory_type == "X" else "X"
    round_max = max(detector["coordinates"][2] for detector in detectors)
    p_type_qubits = {
        qubit_by_xy[tuple(detector["coordinates"][:2])]
        for detector in detectors
        if detector["coordinates"][2] == round_max
        and tuple(detector["coordinates"][:2]) in qubit_by_xy
    }

    for detector_index, detector in enumerate(detectors):
        detector["detector_index"] = detector_index
        detector["qubit"] = qubit_by_xy.get(
            tuple(detector["coordinates"][:2])
        )
        detector["type"] = (
            memory_type if detector["qubit"] in p_type_qubits else other_type
        )

    # Sort detectors by qubit index and round index, with None qubits last
    detectors_sorted = sorted(
        detectors,
        key=lambda detector: (
            detector["qubit"] if detector["qubit"] is not None else float("inf"),
            detector["coordinates"][2],
        ),
    )

   # print(detectors_sorted)

    num_detectors_per_round = code.n - code.k
    total_detectors = detection_events.shape[1]
    num_rounds = total_detectors // num_detectors_per_round
    syndromes = np.zeros_like(
        detection_events.reshape(
            detection_events.shape[0],
            num_rounds,
            num_detectors_per_round,
        ),
        dtype=int,
    )

    # number of shots
    for shot in range(syndromes.shape[0]):
        detection_events_per_shot = detection_events[shot]
        for round_index in range(num_rounds):
            if round_index == 0:
                event_index = 0
                for detector in detectors_sorted:
                    detector_index = detector["detector_index"]
                    if detector["coordinates"][2] != round_index:
                        continue
                    if memory_type == "Z":
                        if event_index < len(p_type_qubits):
                            syndromes[shot, round_index, event_index] = (
                                detection_events_per_shot[detector_index]
                            )
                        else:
                            break
                    elif memory_type == "X":
                        if event_index + len(p_type_qubits) < num_detectors_per_round:
                            syndromes[shot, round_index, event_index + len(p_type_qubits)] = (
                                detection_events_per_shot[detector_index]
                            )
                        elif event_index >= len(p_type_qubits):
                            break
                    event_index += 1
            elif round_index == round_max:
                pass
            else:
                event_index_z = 0
                event_index_x = 0
                for detector in detectors_sorted:
                    detector_index = detector["detector_index"]
                    if detector["coordinates"][2] != round_index:
                        continue
                    if detector["type"] == "Z":
                        syndromes[shot, round_index, event_index_z] = (
                            detection_events_per_shot[detector_index]
                        ) ^ syndromes[shot, round_index - 1, event_index_z]
                        event_index_z += 1
                    elif event_index_x + len(p_type_qubits) < num_detectors_per_round:
                        output_index = event_index_x + len(p_type_qubits)
                        syndromes[shot, round_index, output_index] = (
                            detection_events_per_shot[detector_index]
                        ) ^ syndromes[shot, round_index - 1, output_index]
                        event_index_x += 1

    return syndromes[:, 1:, :]
def count_logical_errors(
    circuit: stim.Circuit,
    num_shots: int
) -> int | tuple[int, np.ndarray]:
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
        The number of shots with a logical error.
    """
    sample = sample_logical_errors(circuit=circuit, num_shots=num_shots)
    return sample.num_errors


def repetition_theory(p: float, d: int) -> float:
    """Compute the theoretical logical error rate of a repetition code."""
    return sum(
        comb(d, j) * p**j * (1 - p)**(d - j)
        for j in range((d + 1) // 2, d + 1)
    )

def Q_alpha(p: float, d: int, alpha: float) -> float:
    pass

class CodeBuilder(ABC):
    """
    Abstract base class for quantum error-correction circuit construction.

    Subclasses must implement the `build_circuit` method.
    """

    @abstractmethod
    def build_code(self, distance: int, noise: dict) -> CodeCircuit:
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
        CodeCircuit
            The constructed circuit and its code parameters.
        """
        pass


#   TODO:
#     - concantenation option (Shor's Code)
#       - build with repetitioncodebuilder

class RepetitionCodeBuilder(CodeBuilder):
    """
    Builder for repetition code circuits.

    Constructs a `[d,1,d]` distance-`d` repetition code with bit-flip noise,
    measurement, detector definitions, and a logical observable. 
    The distance is related to the logical X_L Error. For a phase flip code the phase_flip 
    flag can be set to true. 

    """

    def build_code(self, distance=3, noise={"X_ERROR": 0.05}, logical_one=False, phase_flip=False):
        """
        Build a repetition code circuit.

        Parameters
        ----------
        distance : int, optional
            Code distance. Must be an odd integer ≥ 3.
        noise : dict, optional
            Noise specification. Expected key:
            - "x": float
                Probability of bit-flip (X) error per qubit.
        logical_one : bool, optional
            If True, prepares the logical |1⟩ state instead of |0⟩.
        phase_flip : bool, optional
            If True, the phase-flip repetition code is built.

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

        if phase_flip:
            circuit.append("H", list(range(distance)))
            gate_error = "Z_ERROR"
            measurement = "MX"
            for d in range(1, distance):
                circuit.append("CNOT", [d, 0])
        else:
            gate_error = "X_ERROR"
            measurement = "M"
            for d in range(1, distance):
                circuit.append("CNOT", [0, d])


        circuit.append(gate_error, list(range(distance)), noise[gate_error])

        circuit.append(measurement, list(range(distance)))
        for d in range(1, distance):
            circuit.append(
                "DETECTOR",
                [stim.target_rec(-1 * (d + 1)), stim.target_rec(-1 * d)],
            )

        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

        code = CodeCircuit(
            circuit=circuit,
            n=distance,
            k=1,
            d=distance,
            name="Repetition Code",
            memory_type="X" if phase_flip else "Z"
        )

        return code

class SurfaceCodeBuilder(CodeBuilder):
    def build_code(
        self,
        distance=3,
        noise=0,
        code_task="surface_code:rotated_memory_z",
        rounds=1,
        noise_model="full",
    ):
        """
        Build a surface-code memory experiment circuit.

        Parameters
        ----------
        distance : int
            Code distance.
        noise : float
            Physical error rate to use for the selected noise model.
        code_task : str
            Stim surface-code task name.
        rounds : int
            Number of syndrome extraction rounds.
        noise_model : str
            One of {"full", "bit_flip"}. The ``bit_flip`` model keeps only the
            data-qubit error channel active and turns the other noise channels off.
        """
        if noise_model == "full":
            model = {
                "after_clifford_depolarization": noise,
                "before_round_data_depolarization": noise,
                "before_measure_flip_probability": noise,
                "after_reset_flip_probability": noise,
            }
        elif noise_model == "bit_flip":
            model = {
                "after_clifford_depolarization": 0,
                "before_round_data_depolarization": 0,
                "before_measure_flip_probability": noise,
                "after_reset_flip_probability": noise,
            }
        else:
            raise ValueError(
                "noise_model must be one of {'full', 'bit_flip'}; "
                f"got {noise_model!r}."
            )

        circuit = generate_circuit(
            code_task=code_task,
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=model.get(
                "after_clifford_depolarization"),
            before_round_data_depolarization=model.get(
                "before_round_data_depolarization"),
            before_measure_flip_probability=model.get(
                "before_measure_flip_probability"),
            after_reset_flip_probability=model.get(
                "after_reset_flip_probability"),
        )

        return CodeCircuit(
            circuit=circuit,
            name=code_task,
            memory_type="X" if "x" in code_task else "Z",
            n=distance ** 2,
            k=1,
            d=distance,
        )

class SinisterSimulation:
    """
    Driver class for running quantum error-correction simulations.

    This class generates parameter sweeps over code distances and
    physical error rates, executes simulations using Sinter, and
    provides visualization utilities.
    """

    def __init__(
        self,
        code_builder,
        distances: Iterable[int] = (3, 5, 7, 9),
        noise_values: Iterable[float] = (0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5),
        num_workers: int = 4,
        max_shots: int = 100_000,
        max_errors: int = 500,
        rounds: int = 3,
        code_task: str = "surface_code:rotated_memory_x",
        noise_model: str = "full",
    ):
        """
        Initialize simulation parameters.

        Parameters
        ----------
        code_builder : CodeBuilder
            Instance responsible for constructing codes.
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
        self.code_builder = code_builder
        self.distances = distances
        self.noise_values = noise_values
        self.num_workers = num_workers
        self.max_shots = max_shots
        self.max_errors = max_errors
        self.rounds = rounds
        self.code_task = code_task
        self.noise_model = noise_model

    def _build_code(self, d: int, p: float) -> CodeCircuit:
        """Construct a code for the requested distance/noise pair."""
        try:
            return self.code_builder.build_code(
                distance=d,
                noise=p,
                rounds=self.rounds,
                code_task=self.code_task,
                noise_model=self.noise_model,
            )
        except TypeError:
            try:
                return self.code_builder.build_code(
                    distance=d,
                    noise=p,
                    rounds=self.rounds,
                    code_task=self.code_task,
                    noise_model=self.noise_model,
                )
            except TypeError:
                try:
                    return self.code_builder.build_code(
                        distance=d,
                        noise=p,
                        rounds=self.rounds,
                        code_task=self.code_task,
                        noise_model=self.noise_model,
                    )
                except TypeError:
                    return self.code_builder.build_code(d, p)

    def run_single(self, d: int, p: float) -> float:
        """
        Execute a single logical error rate simulation.

        A code is constructed for the specified code distance and
        physical error rate, simulated for `max_shots` samples, and
        decoded using PyMatching. The logical error rate (LER) is
        computed as the fraction of shots resulting in a logical failure.

        Parameters
        ----------
        d : int
            Code distance.
        p : float
            Physical error rate.

        Returns
        -------
        float
            Estimated logical error rate per shot.
        """

        num_errors = count_logical_errors(
            circuit=self._build_code(d, p).circuit,
            num_shots=self.max_shots,
        )

        return num_errors / self.max_shots

    def syndrome_resample_single(
        self,
        d: int,
        p: float,
        alpha: float = 2.0,
    ) -> SyndromeResamplingResult:
        """Run one syndrome-resampling estimate for a single distance/noise pair."""
        code = self._build_code(d, p)
        return syndrome_resample_logical_errors(
            circuit=code,
            num_shots=self.max_shots,
            alpha=alpha,
        )

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

    def run_alpha_sweep(
        self,
        alphas: Iterable[float] = (1.0, 2.0, 3.0),
        noise_values: Iterable[float] | None = None,
        distances: Iterable[int] | None = None,
    ) -> np.ndarray:
        """
        Evaluate the logical error rate under several resampling exponents.

        Returns an array of shape ``(len(distances), len(alphas), len(noise_values))``
        where each entry is the resampled logical error rate for a given distance,
        alpha exponent, and physical error rate.
        """
        if noise_values is None:
            noise_values = self.noise_values
        if distances is None:
            distances = self.distances

        alphas = tuple(float(alpha) for alpha in alphas)
        noise_values = tuple(float(p) for p in noise_values)
        distances = tuple(int(d) for d in distances)

        result = np.empty((len(distances), len(alphas), len(noise_values)), dtype=float)

        for i, d in enumerate(distances):
            for j, alpha in enumerate(alphas):
                for k, p in enumerate(noise_values):
                    code = self._build_code(d, p)
                    value = syndrome_resample_logical_errors(
                        circuit=code,
                        num_shots=self.max_shots,
                        alpha=alpha,
                    )
                    result[i, j, k] = value.resampled_logical_error_rate

        return result

    def plot_alpha_sweep(
        self,
        resampled_rates: np.ndarray,
        alphas: Iterable[float] = (1.0, 2.0, 3.0),
        noise_values: Iterable[float] | None = None,
        distances: Iterable[int] | None = None,
        title: str = "Surface Code",
    ):
        """Plot one subplot per alpha, with distance as the curve family in each panel."""
        if noise_values is None:
            noise_values = self.noise_values
        if distances is None:
            distances = self.distances

        alphas = tuple(float(alpha) for alpha in alphas)
        noise_values = tuple(float(p) for p in noise_values)
        distances = tuple(int(d) for d in distances)

        markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

        fig, axes = plt.subplots(
            1,
            len(alphas),
            figsize=(4 * len(alphas) + 2, 5),
            sharex=True,
            sharey=True,
        )

        if len(alphas) == 1:
            axes = [axes]

        for j, alpha in enumerate(alphas):
            ax = axes[j]
            for i, d in enumerate(distances):
                ax.plot(
                    noise_values,
                    resampled_rates[i, j],
                    marker=markers[i % len(markers)],
                    linewidth=2,
                    markersize=6,
                    label=f"d={d}",
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(min(noise_values), max(noise_values))

            finite = resampled_rates[:, j][resampled_rates[:, j] > 0]
            if finite.size > 0:
                ax.set_ylim(max(np.min(finite) / 2, 1e-6), 1)
            else:
                ax.set_ylim(1e-6, 1)

            ax.set_title(f"alpha = {alpha}")
            ax.grid(which="major")
            ax.grid(which="minor", alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"{title} syndrome-resampled LER")
        fig.supxlabel("Physical Error Rate")
        fig.supylabel("Logical Error Rate per Shot")
        fig.set_dpi(120)
        plt.tight_layout()
        plt.show()

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