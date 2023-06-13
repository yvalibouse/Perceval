import numpy as np
import perceval as pcvl
import networkx as nx


def compute(input_state, basis, s):
    r"""Compute the output state of a simulator with a given input_state and its computional basis.
    If give_ampli is True, it also returns the list of amplitude that can be interpreted by qsphere from qiskit
    """
    l = len(basis)

    output_state = pcvl.StateVector()

    for state in input_state:
        for i in range(l):
            a = s.probampli(state, basis[i])
            output_state += a * pcvl.StateVector(basis[i])

    return output_state


def logical(sv):
    r"""This function returns the list of the logical n-qubits of the StateVector with dual rail encoding and raises an error if the StateVector doesn't represent a logical n-qubits
    """
    l_sv = len(sv)
    if l_sv == 0:
        raise ValueError("The StateVector is empty, too bad :p")
    l_bs = len(sv[0])
    if l_bs % 2 != 0:
        raise ValueError("The StateVector doesn't represent a logical n-qubit")
    else:
        l_n_qbt = l_bs // 2

    logbit_list = []
    ampli_list = []
    for state in sv:
        bs = pcvl.BasicState(state)
        log_state = []
        for i in range(l_n_qbt):
            # check the value of each qubit
            # i-th qubit = 0
            if bs[2 * i] == 1 and bs[2 * i + 1] == 0:
                log_state.append(0)
                # i-th qubit = 1
            elif bs[2 * i] == 0 and bs[2 * i + 1] == 1:
                log_state.append(1)
            else:
                raise ValueError("The StateVector doesn't represent a n-qubit")
        ampli_list.append(sv[bs])
        logbit_list.append(np.array(log_state))
    return logbit_list, ampli_list


def is_graph_state(sv):
    r"""Compute the graph if self is a graph state, raise an error otherwise
    """
    log_list, ampli_list = logical(sv)
    l = len(log_list)
    n = len(log_list[0])  # number of vertices

    if l != 2 ** n:
        raise ValueError("The state isn't a graph state: not enough terms in this StateVector")

    edges = []
    tocheck_list = []
    amp0 = ampli_list[0]

    for i in range(l):
        a_i = ampli_list[i]
        state_i = log_list[i]
        if np.imag(a_i) != 0:
            raise ValueError("The state isn't a graph state: complex amplitude")
        a_i = np.real(a_i)
        if abs(a_i) != abs(amp0):  # all the states need to have the same probability
            raise ValueError("The state isn't a graph state: the absolute values of the amplitudes are different")
        vert = np.where(state_i == 1)[0]
        if len(vert) == 2:
            if a_i < 0:
                edges.append((vert[0], vert[1]))
        else:
            tocheck_list.append((a_i, state_i))

    if edges == []:
        for (amp, state) in tocheck_list:
            if (amp0 > 0) & (amp < 0):
                raise ValueError("The state isn't a graph state: the signs doesn't match")
            if (amp0 < 0) & (amp > 0):
                raise ValueError("The state isn't a graph state: the signs doesn't match")
    for (amp, state) in tocheck_list:  # verify the sign flip for each CZ applied
        sign = int(amp / abs(amp))
        sign_test = 1
        for edge in edges:
            if state[edge[0]] == 1 and state[edge[1]] == 1:
                sign_test *= -1
        if sign_test != sign:
            raise ValueError("The state isn't a graph state: the signs doesn't match")

    g = nx.Graph()
    g.add_nodes_from([i for i in range(n)])
    g.add_edges_from(edges)
    return g

