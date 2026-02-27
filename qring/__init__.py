"""
QRiNG: Quantum Random Number Generator Protocol

A hybrid quantum-blockchain protocol for verifiable quantum random number
generation using Ethereum smart contracts and Quantum Key Distribution (QKD).

Author: Jeffrey Morais, BTQ
"""

from qring.simulator import QRiNGSimulator
from qring.emulator import QRiNGEmulator
from qring.visualization import QRiNGVisualizer

__version__ = "0.1.0"
__author__ = "Jeffrey Morais"

__all__ = [
    "QRiNGSimulator",
    "QRiNGEmulator",
    "QRiNGVisualizer",
]
