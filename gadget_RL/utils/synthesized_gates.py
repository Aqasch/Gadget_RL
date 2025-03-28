from qiskit import *
import random

def rzcz(theta, ctrl, targ):
    '''
    ------[RZ(ctrl)]----o(ctrl)----
                        |
                        |
    --------------------o----------
    '''
    circ = QuantumCircuit(2, name=f"rzcz{ctrl}{targ}")
    circ.rz(theta, 0)
    circ.cz(0, 1)
    circ = circ.to_gate()
    return circ

def rzcz_noise(theta, ctrl, targ):
    '''
    ------[RZ(ctrl)]----o(ctrl)----
                        |
                        |
    --------------------o----------
    '''
    circ = QuantumCircuit(2, name=f"rzcz{ctrl}{targ}")
    circ.rz(theta, 0)
    circ.cz(0, 1)
    circ = circ.to_gate()

    return circ


def rz0rz1cz01(theta, ctrl, targ):
    '''
    ------[RZ(theta)]---o(ctrl)----
                        |
                        |
    ------[RZ(theta)]---o(targ)----
    '''
    circ = QuantumCircuit(2, name=f"rzrzcz{ctrl}{targ}")
    circ.rz(theta, 0)
    circ.rz(theta, 1)
    circ.cz(0, 1)
    circ = circ.to_gate()
    return circ

def xrz(theta):
    '''
    ------[X]--RZ(theta)-----
    '''
    circ = QuantumCircuit(1, name="xrz_gate")
    circ.x(0)
    circ.rz(theta, 0)
    circ = circ.to_gate()
    return circ


def ryrx(theta, qubitry, qubitrx):
    '''
    ------[RY(theta)]--RX(theta)-----
    '''
    circ = QuantumCircuit(2, name="ryrx_gate")
    circ.ry(theta, 0)
    circ.rx(theta, 1)
    circ = circ.to_gate()
    return circ

def rxry(theta, qubitrx, qubitry):
    '''
    ------[RY(theta)]--RX(theta)-----
    '''
    circ = QuantumCircuit(2, name="rxry_gate")
    circ.rx(theta, 0)
    circ.ry(theta, 1)
    circ = circ.to_gate()
    return circ

def xsx():
    '''
    ------[Sqrt(X)]----------
    
    ------[X]------[Sqrt(X)]--
    '''
    circ = QuantumCircuit(1, name="xsx_gate")
    circ.x(0)
    circ.sx(0)
    circ = circ.to_gate()
    return circ


