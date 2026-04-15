"""
QRiNG Example: Basic Simulation

Runs the QRiNG simulator and prints the consensus results.
"""

from qring import QRiNGSimulator

sim = QRiNGSimulator(numNodes=6, bitstringLength=8, seed=42)
results = sim.runFullSimulation()

print(f"Honest nodes: {results['honest_nodes']}")
print(f"Final random bits: {results['final_random_number']}")
