from simulation import RepetitionCodeBuilder
from math import comb

def quantum_hamming_bound(n, k, d, num_P=3):
    # num_P = number of correctable Pauli Errors usually 3. In the case of the repetition code it is 1
    t = (d - 1) // 2
    lhs = sum(comb(n, j) * (num_P)**j for j in range(t + 1)) * 2**k
    rhs = 2**n
    return lhs <= rhs

def test_1():
    circuit_builder = RepetitionCodeBuilder()
    for d in [3, 5, 7, 9]:
        circuit = circuit_builder.build_circuit(distance=d)
        dem = circuit.detector_error_model()
        assert dem.num_detectors ==  circuit.num_qubits - dem.num_observables

def test_2():
    circuit_builder = RepetitionCodeBuilder()
    for d in [3, 5, 7, 9]:
        circuit = circuit_builder.build_circuit(distance=d)
        dem = circuit.detector_error_model()
        n = circuit.num_qubits
        k = dem.num_observables

        # repetition code only corrects one Pauli Error
        assert quantum_hamming_bound(n,k,d,num_P=1)

def test_3():
    circuit_builder = RepetitionCodeBuilder()
    circuit = circuit_builder.build_circuit()    
    for d in [3, 5, 7, 9]:
        circuit = circuit_builder.build_circuit(distance=d)    
        dem = circuit.detector_error_model()
    err = dem.shortest_graphlike_error()
    distance = len(err)
    # Short Comment about fault distance and stim detectors

    assert distance == d

test_1()
test_2()
test_3()