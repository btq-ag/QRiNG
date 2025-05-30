# QRiNG Code Implementation

## Overview

This directory contains the complete implementation of the **QRiNG (Quantum Random Number Generator)** protocol, including smart contract code, simulation tools, and visualization components. The QRiNG system combines quantum key distribution (QKD) with blockchain technology to generate cryptographically secure random numbers through decentralized consensus.

## File Structure

### Core Implementation Files

| File | Description | Type | Lines of Code |
|------|-------------|------|---------------|
| [`originalQRiNG.sol`](./originalQRiNG.sol) | Smart contract implementation in Solidity | Blockchain | ~200 LOC |
| [`simulatorQRiNG.py`](./simulatorQRiNG.py) | QRiNG protocol simulator with quantum modeling | Python | ~590 LOC |
| [`emulatorQRiNG.py`](./emulatorQRiNG.py) | Smart contract emulator for testing | Python | ~655 LOC |
| [`visualizationQRiNG.py`](./visualizationQRiNG.py) | Animated visualization suite | Python | ~920 LOC |

### Example Files

| File | Description | Purpose |
|------|-------------|---------|
| [`exampleAnimation.py`](./exampleAnimation.py) | Example animation creation | Tutorial/Demo |
| [`exampleStatic.py`](./exampleStatic.py) | Example static visualization | Tutorial/Demo |

## Architecture Overview

### 1. Smart Contract Layer (`originalQRiNG.sol`)

**Core Components:**
- **Voter Struct**: Manages participant information and quantum bitstrings
- **Consensus Mechanism**: Validates nodes through bitstring similarity comparison
- **Random Number Generation**: XOR aggregation of honest nodes' bitstrings
- **Access Control**: Admin-only functions with proper modifiers

**Key Functions:**
```solidity
function addNewString(uint[][] memory newString) public
function setAddresses(address[] memory voterAddresses) public
function startVoting() external onlyAdmin
function consensusCheck() external
function randomNumber() external view returns (uint[] memory)
```

**Gas Optimization Features:**
- Efficient storage patterns
- Minimal computational overhead
- Event-driven architecture for off-chain monitoring

### 2. Simulation Layer (`simulatorQRiNG.py`)

**Quantum Modeling:**
- **Bell State Simulation**: Models quantum entanglement between nodes
- **Measurement Uncertainty**: Simulates quantum measurement with probabilistic outcomes
- **Decoherence Effects**: Includes realistic quantum noise and decay
- **Correlation Modeling**: Simulates entanglement correlations in bitstring generation

**Consensus Protocol:**
- **Peer Validation**: Each node validates others through bitstring comparison
- **Threshold-Based Honesty**: Nodes exceeding similarity threshold considered honest
- **Byzantine Fault Tolerance**: Resistant to malicious node behavior
- **Deterministic Randomness**: Final output through XOR of honest nodes

**Network Simulation:**
```python
class QRiNGSimulator:
    def _generate_quantum_bitstrings(self)    # Quantum key distribution simulation
    def perform_consensus_check(self, node)   # Individual node validation
    def run_consensus_protocol(self)          # Full network consensus
    def generate_final_random_number(self)    # XOR aggregation
```

### 3. Emulation Layer (`emulatorQRiNG.py`)

**Smart Contract Emulation:**
- **State Management**: Maintains blockchain-like state without actual deployment
- **Gas Calculation**: Simulates realistic gas consumption patterns
- **Event Emission**: Tracks contract events for debugging and monitoring
- **Transaction Logging**: Complete audit trail of all operations

**Testing Features:**
- **End-to-End Testing**: Full protocol execution simulation
- **Error Handling**: Comprehensive exception management
- **Performance Metrics**: Gas usage and execution time tracking
- **State Inspection**: Real-time contract state monitoring

**Emulation Capabilities:**
```python
class QRiNGEmulator:
    def add_new_string(self, bitstrings, caller)     # Store quantum data
    def set_addresses(self, addresses, caller)       # Register participants
    def start_voting(self, caller)                   # Begin consensus
    def consensus_check(self, caller)                # Execute validation
    def random_number(self)                          # Generate final result
```

### 4. Visualization Layer (`visualizationQRiNG.py`)

**Animation Suite:**
- **QKD Process Animation**: Shows quantum key distribution between nodes
- **Consensus Mechanism**: Visualizes node validation and voting process
- **Smart Contract Execution**: Displays blockchain interaction flow
- **Real-time Metrics**: Live updating statistics and measurements

**Visualization Types:**
1. **Network Topology**: Node relationships and quantum channels
2. **Quantum State Evolution**: Time-domain quantum amplitude plots
3. **Bitstring Matrices**: Similarity heatmaps and comparison tables
4. **Protocol Flow**: Step-by-step process diagrams with mathematical formulas

## Installation & Dependencies

### Required Python Packages

```bash
pip install numpy matplotlib networkx seaborn pillow
```

### Package Versions (Tested)
- `numpy >= 1.21.0` - Numerical computations and quantum modeling
- `matplotlib >= 3.5.0` - Plotting and animation generation
- `networkx >= 2.6.0` - Network graph visualization
- `seaborn >= 0.11.0` - Statistical visualization enhancements
- `pillow >= 8.3.0` - Image processing for GIF generation

### Solidity Environment
- **Solidity Version**: `^0.8.0`
- **Framework**: Compatible with Truffle, Hardhat, or Remix
- **Network**: Ethereum, Polygon, or any EVM-compatible blockchain

## Usage Instructions

### 1. Running the Simulator

```python
# Basic simulation execution
python simulatorQRiNG.py

# Custom parameters
simulator = QRiNGSimulator(num_nodes=6, bitstring_length=10, seed=42)
results = simulator.run_full_simulation()
```

**Output:**
- Network consensus results
- Honest node identification
- Final random number generation
- Statistical analysis plots

### 2. Smart Contract Emulation

```python
# End-to-end emulation
python emulatorQRiNG.py

# Custom testing
emulator = QRiNGEmulator()
emulator.add_new_string(quantum_data, admin_address)
emulator.set_addresses(voter_addresses, admin_address)
final_random = emulator.random_number()
```

**Features:**
- Gas usage tracking
- Transaction logging
- Event monitoring
- State inspection

### 3. Visualization Generation

```python
# Generate all animations
python visualizationQRiNG.py

# Custom visualizations
visualizer = QRiNGVisualizer()
visualizer.animate_qkd_process("qkd_animation.gif")
visualizer.animate_consensus_mechanism("consensus_animation.gif")
```

**Output Files:**
- `qkd_process.gif` - Quantum key distribution animation
- `consensus_mechanism.gif` - Node validation process
- `smart_contract_execution.gif` - Blockchain interaction flow

## Technical Specifications

### Quantum Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `num_nodes` | 4-6 | Number of participating nodes |
| `bitstring_length` | 6-10 | Length of quantum bitstrings |
| `quantum_bias` | 0.5 ± 0.05 | Quantum measurement bias |
| `correlation_factor` | 0.15 | Entanglement correlation strength |
| `decoherence_rate` | 1/15 | Quantum state decay rate |

### Consensus Thresholds

| Threshold | Formula | Purpose |
|-----------|---------|---------|
| Similarity | `matches > bitstring_length / 2` | Node validation |
| Honesty | `vote_count > num_nodes / 2` | Honest node identification |
| Network | `honest_nodes >= 1` | Protocol execution |

### Gas Consumption (Estimated)

| Function | Base Gas | Variable Cost | Total (Typical) |
|----------|----------|---------------|-----------------|
| `addNewString` | 21,000 | 20 per bit | ~25,000 |
| `setAddresses` | 21,000 | 50,000 per voter | ~321,000 |
| `consensusCheck` | 21,000 | 5,000 per comparison | ~51,000 |
| `randomNumber` | 21,000 | 1,000 per XOR | ~27,000 |

## Security Considerations

### Quantum Security
- **True Randomness**: Quantum measurement provides genuine entropy
- **Entanglement Verification**: Correlation checks prevent classical spoofing
- **Decoherence Modeling**: Realistic quantum noise simulation

### Blockchain Security
- **Access Control**: Admin-only functions with proper modifiers
- **Input Validation**: Comprehensive parameter checking
- **Reentrancy Protection**: State updates before external calls
- **Integer Overflow**: SafeMath equivalent checks

### Network Security
- **Byzantine Tolerance**: Handles up to (n-1)/3 malicious nodes
- **Consensus Validation**: Multiple verification rounds
- **Threshold Security**: Configurable security parameters

## Performance Metrics

### Computational Complexity
- **Quantum Generation**: O(n × l) where n = nodes, l = bitstring length
- **Consensus Protocol**: O(n²) for all pairwise comparisons
- **Random Generation**: O(n) for XOR aggregation
- **Visualization**: O(n² × f) where f = animation frames

### Memory Usage
- **Bitstring Storage**: n × l × 8 bytes
- **Vote Tracking**: n × 4 bytes
- **Transaction Log**: Variable (growth over time)
- **Visualization Buffer**: ~50MB per animation

## Development Guidelines

### Code Style
- **Python**: PEP 8 compliance with comprehensive docstrings
- **Solidity**: Solidity Style Guide with NatSpec documentation
- **Comments**: One-line explanatory comments for complex logic
- **Attribution**: "Author: Jeffrey Morais, BTQ" in all file headers

### Testing Protocol
1. **Unit Tests**: Individual function validation
2. **Integration Tests**: Full protocol execution
3. **Performance Tests**: Gas usage and execution time
4. **Security Tests**: Attack vector validation

### Contribution Workflow
1. Fork repository and create feature branch
2. Implement changes with comprehensive comments
3. Add unit tests for new functionality
4. Update documentation and README files
5. Submit pull request with detailed description

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**Animation Generation Failures:**
```python
# Reduce animation complexity
visualizer = QRiNGVisualizer()
visualizer.animate_qkd_process("output.gif", frames=30, dpi=75)
```

**Memory Issues:**
```python
# Reduce simulation parameters
simulator = QRiNGSimulator(num_nodes=4, bitstring_length=6)
```

**Gas Estimation Errors:**
- Check input parameter sizes
- Verify network connection
- Update gas price estimates

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
- [ ] Multi-threaded simulation for large networks
- [ ] Real quantum hardware integration
- [ ] Advanced visualization with 3D rendering
- [ ] Smart contract optimization for Layer 2 networks
- [ ] Machine learning analysis of consensus patterns

### Research Directions
- [ ] Post-quantum cryptographic integration
- [ ] Scalability analysis for enterprise deployment
- [ ] Cross-chain random number sharing
- [ ] Quantum advantage verification protocols

## License & Attribution

**Author**: Jeffrey Morais, BTQ  
**License**: [Specify License]  
**Institution**: BTQ Technologies  

For questions, issues, or collaboration inquiries, please contact the development team.

---

*This implementation represents a proof-of-concept for quantum-enhanced random number generation in decentralized systems. All quantum simulations are classical approximations for demonstration purposes.*
