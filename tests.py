from simulation import RepetitionCodeBuilder
from math import comb

# Distance == min weight stabilizers

def test_1():
    circuit_builder = RepetitionCodeBuilder()
    for d in [3, 5, 7, 9]:
        circuit = circuit_builder.build_circuit(distance=d)
        dem = circuit.detector_error_model()
        print(dem.num_detectors+dem.num_observables)

test_1()

def quantum_hamming_bound(n, k, d):
    t = (d - 1) // 2
    lhs = sum(comb(n, j) * 3**j for j in range(t + 1)) * 2**k
    rhs = 2**n
    return lhs <= rhs

def test_2():
    circuit_builder = RepetitionCodeBuilder()
    for d in [3, 5, 7, 9]:
        circuit = circuit_builder.build_circuit(distance=d)
        dem = circuit.detector_error_model()
        n = circuit.num_qubits
        k = dem.num_observables
        assert quantum_hamming_bound(n,k,d)