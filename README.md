# QRiNG: Quantum Random Number Generator Protocol
###### On-chain consensus over quantum-generated bitstrings, verified by majority vote.

<p align="center">
  <a href="https://github.com/btq-ag/QRiNG/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/btq-ag/QRiNG/ci.yml?branch=main&label=CI&logo=github" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/install-pip%20install%20--e%20.%5Bdev%5D-brightgreen" alt="pip install"></a>
</p>

![QKD Process Animation](Logos/QRiNG_extended_dark.png)

## Objective

QRiNG generates verifiable random numbers by combining quantum measurement with Ethereum smart contract consensus. Participants submit Hadamard-circuit bitstrings to a Solidity contract, validators run pairwise similarity checks, and honest nodes are identified by majority vote. The final output is the XOR aggregate of all honest bitstrings. 58 tests across simulator and emulator, CI on Python 3.10, 3.11, 3.12.

## Theoretical Background

For a qubit in equal superposition $|\psi\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$, measurement yields each outcome with probability $P(0) = P(1) = 1/2$ (Born's rule), giving maximum entropy $H = 1$ bit per qubit. When `qiskit-aer` is installed, bitstrings are generated via Hadamard + measure circuits on `AerSimulator`; otherwise a seeded PRNG fallback is used.

Consensus works by pairwise bitstring comparison. For bitstrings $\mathbf{b}_i, \mathbf{b}_j$ of length $\ell$, the similarity score counts matching positions:

$$S(\mathbf{b}_i, \mathbf{b}_j) = \ell - d_H(\mathbf{b}_i, \mathbf{b}_j)$$

A node is classified as honest when $\text{VoteCount}(i) > |V|/2$. The final random output XORs all honest bitstrings:

$$R[k] = \bigoplus_{i \in V_{\text{honest}}} b_i^{(k)} \pmod{2}$$

If at least one honest node contributes true quantum randomness, the output maintains full entropy.

## Installation

```bash
pip install -e ".[dev]"   # includes Qiskit Aer, pytest, ruff, mypy
```

## Code Functionality

### Quantum Simulation

The simulator generates per-node bitstrings via Hadamard circuits, runs pairwise consensus, and XOR-aggregates honest outputs:

```python
from qring import QRiNGSimulator

sim = QRiNGSimulator(numNodes=6, bitstringLength=8, seed=42)
results = sim.runFullSimulation()
print(results["honest_nodes"], results["final_random_number"])
```

Key methods: `calculateBitstringSimilarity(node1, node2)`, `performConsensusCheck(checkingNode)`, `generateFinalRandomNumber()`.

### Smart Contract Emulation

The emulator replicates the Solidity contract logic in Python, including admin-only access control, one-time initialization guards, gas estimation, and event emission:

```python
from qring import QRiNGEmulator

emu = QRiNGEmulator(bitstringLength=6, adminAddress="0xADMIN")
emu.addNewString([[1,0,1,1,0,1], [1,1,1,0,0,1]], "0xADMIN")
emu.setAddresses(["0xA", "0xB"], "0xADMIN")
for i, addr in enumerate(["0xA", "0xB"]):
    emu.check(i, addr)
emu.endVoting("0xADMIN")
print(emu.randomNumber("0xADMIN"))
```

### Visualization

`QRiNGVisualizer` generates animated GIFs covering the QKD pipeline, consensus voting, and contract execution flow. See [`qring/visualization.py`](qring/visualization.py).

### Solidity Contract

The on-chain contract lives at [`contracts/originalQRiNG.sol`](contracts/originalQRiNG.sol). Admin is set at deployment; access control modifiers protect `addNewString` and `setAddresses`; a one-time initialization guard prevents re-registration attacks. A configurable `consensusThreshold` allows tuning correlation requirements before initialization.

## Results

### Quantum Measurement Process

![QKD Process Animation](Plots/qkd_process.gif)

Qubits are initialized in equal superposition, measured via Hadamard + measure circuits on `AerSimulator`. The animation shows the full pipeline from state preparation through bitstring extraction.

### Smart Contract Execution

![Smart Contract Execution](Plots/smart_contract_execution.gif)

Full transaction lifecycle: bitstring upload, voter registration, pairwise consensus checks, voting termination, and random number extraction. Gas costs follow the Ethereum model (base 21,000 + per-voter and per-bit storage costs).

### Network Consensus

![Quantum Network](Plots/qring_simulator_network.png)

Consensus results for a 6-node simulation. Green nodes passed majority-vote validation ($\text{VoteCount} > n/2$); red nodes did not. The heatmap shows pairwise similarity scores.

### Bitstring Analysis

![Quantum States](Plots/qring_simulator_quantum.png)

Per-node bit frequency, quantum state evolution, and XOR aggregation breakdown.

### Emulator Execution

![Contract Execution](Plots/qring_emulator_execution.png)

Gas consumption per function call, voter state matrix, similarity heatmap, and contract event log.

## Project Structure

```
qring/
  __init__.py          # Package exports
  simulator.py         # QKD simulation + consensus
  emulator.py          # Solidity contract emulation
  visualization.py     # Animated GIF generation
contracts/
  originalQRiNG.sol    # Ethereum smart contract
tests/
  test_simulator.py    # 27 simulator tests
  test_emulator.py     # 31 emulator tests
examples/
  exampleSimulation.py # Simulation demo
  exampleEmulation.py  # Emulator demo
```

## Next Steps

- [ ] Integrate with quantum hardware via cloud APIs (IBM Quantum)
- [ ] Add error correction for noisy quantum channels
- [ ] Gas optimization and layer-2 scaling for larger networks
- [ ] Formal verification of the Solidity contract

---

*© 2024-2026 BTQ Technologies, MIT License*
