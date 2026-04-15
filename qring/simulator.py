"""
QRiNG Simulator - Quantum Random Number Generator Simulation

This script simulates the QRiNG protocol by modeling:
1. Quantum Key Distribution (QKD) between network nodes
2. Consensus mechanism for node validation
3. Final random number generation through XOR aggregation
4. Comprehensive visualizations of the entire process

When qiskit-aer is installed, bitstrings are generated from genuine
Hadamard-gate circuits measured on AerSimulator. Otherwise, a classical
PRNG fallback is used (clearly labeled).

Author: Jeffrey Morais, BTQ
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import seaborn as sns
import os
from datetime import datetime

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False

class QRiNGSimulator:
    """
    Simulator for the QRiNG (Quantum Random Number Generator) protocol
    """
    
    def __init__(self, numNodes: int = 6, bitstringLength: int = 8,
                 seed: int | None = None, useQuantumBackend: bool | None = None,
                 consensusThreshold: int | None = None) -> None:
        """
        Initialize the QRiNG simulator
        
        Args:
            numNodes (int): Number of nodes in the network
            bitstringLength (int): Length of quantum bitstrings
            seed (int): Random seed for reproducibility
            useQuantumBackend (bool | None): If True, require Qiskit Aer.
                If None (default), auto-detect: use Qiskit when available.
            consensusThreshold (int | None): Minimum matching bits for a node
                to be considered honest. Defaults to bitstringLength // 2.
        """
        self.rng = np.random.default_rng(seed)
        
        if useQuantumBackend is True and not _HAS_QISKIT:
            raise RuntimeError("qiskit-aer is required for the quantum backend. "
                               "Install with: pip install qiskit qiskit-aer")
        self.useQuantumBackend = _HAS_QISKIT if useQuantumBackend is None else useQuantumBackend
        
        self.numNodes = numNodes
        self.bitstringLength = bitstringLength
        self.consensusThreshold = consensusThreshold if consensusThreshold is not None else bitstringLength // 2
        self.nodes = list(range(numNodes))
        self.bitstrings = {}
        self.voteCounts = np.zeros(numNodes)
        self.hasVoted = np.zeros(numNodes, dtype=bool)
        self.honestNodes = []
        self.finalRandomBits = None
        
        # Generate quantum bitstrings through simulated QKD
        self._generateQuantumBitstrings()
        
    def _generateQuantumBitstrings(self) -> None:
        """
        Generate bitstrings for each node, dispatching to the quantum
        backend (Qiskit Aer) when available or the classical fallback.
        """
        if self.useQuantumBackend:
            self._generateQiskitBitstrings()
        else:
            self._generateClassicalBitstrings()

    def _generateQiskitBitstrings(self) -> None:
        """
        Generate bitstrings via Hadamard-gate circuits on AerSimulator.
        Each node gets one independent circuit of `bitstringLength` qubits,
        all placed in equal superposition and measured once.
        """
        print("Generating quantum bitstrings via Qiskit Aer...")
        sim = AerSimulator(seed_simulator=int(self.rng.integers(0, 2**31)))

        for node in self.nodes:
            qc = QuantumCircuit(self.bitstringLength, self.bitstringLength)
            qc.h(range(self.bitstringLength))
            qc.measure(range(self.bitstringLength), range(self.bitstringLength))
            result = sim.run(qc, shots=1).result()
            bitstring_str = list(result.get_counts().keys())[0]
            # Qiskit returns bitstrings in big-endian order
            self.bitstrings[node] = np.array([int(b) for b in bitstring_str], dtype=int)

        print(f"Generated {len(self.bitstrings)} quantum bitstrings (Qiskit Aer)")

    def _generateClassicalBitstrings(self) -> None:
        """
        Classical PRNG fallback. Uses numpy default_rng to produce bitstrings.
        NOTE: This does NOT provide true quantum randomness.
        """
        print("Generating classical bitstrings (PRNG fallback)...")

        for node in self.nodes:
            # Simulate quantum measurement outcomes
            # Each bit has quantum uncertainty with bias towards true randomness
            quantum_bias = 0.5 + self.rng.normal(0, 0.05)  # Slight deviation from perfect 50/50
            quantum_bias = np.clip(quantum_bias, 0.3, 0.7)  # Keep within reasonable bounds
              # Generate bitstring with quantum-like properties
            bitstring = []
            for bit_idx in range(self.bitstringLength):
                # Simulate quantum measurement with entanglement correlations
                if bit_idx > 0:
                    # Add correlation with previous bit (simulating entanglement)
                    correlation_factor = 0.15 * np.sin(bit_idx * np.pi / 4)
                    prob = quantum_bias + correlation_factor * (2 * bitstring[-1] - 1)
                    prob = np.clip(prob, 0.1, 0.9)  # Ensure valid probability range
                else:
                    prob = quantum_bias
                
                # Generate quantum bit based on calculated probability
                bit = 1 if self.rng.random() < prob else 0
                bitstring.append(bit)
            
            self.bitstrings[node] = np.array(bitstring)
            
        print(f"Generated {len(self.bitstrings)} classical bitstrings (PRNG fallback)")
    
    def calculateBitstringSimilarity(self, node1: int, node2: int) -> int:
        """
        Calculate similarity between two nodes' bitstrings
        Implements the matching function from the smart contract
        """
        if node1 not in self.bitstrings or node2 not in self.bitstrings:
            return 0
        
        matches = np.sum(self.bitstrings[node1] == self.bitstrings[node2])
        return matches
    
    def performConsensusCheck(self, checking_node: int) -> bool:
        """
        Simulate the consensus check function from the smart contract
        Each node checks all other nodes' bitstrings
        """
        if self.hasVoted[checking_node]:
            return False
        
        threshold = self.consensusThreshold
        
        for target_node in self.nodes:
            if target_node != checking_node:
                # Calculate similarity between bitstrings
                matches = self.calculateBitstringSimilarity(checking_node, target_node)
                # If similarity exceeds threshold, increment vote count
                if matches > threshold:
                    self.voteCounts[target_node] += 1
        
        # Mark this node as having voted to prevent double-voting
        self.hasVoted[checking_node] = True
        return True
    
    def runConsensusProtocol(self) -> list[int]:
        """
        Run the complete consensus protocol with all nodes participating
        """
        print("Running consensus protocol...")
        
        # Each node performs checking
        for node in self.nodes:
            self.performConsensusCheck(node)
          # Determine honest nodes (those with vote count > numNodes/2)
        threshold = len(self.nodes) // 2
        self.honestNodes = [node for node in self.nodes if self.voteCounts[node] > threshold]
        
        print(f"Honest nodes identified: {self.honestNodes}")
        return self.honestNodes
    
    def generateFinalRandomNumber(self) -> npt.NDArray[np.int_] | None:
        """
        Generate final random number by XOR-ing honest nodes' bitstrings
        Implements the randomNumber() function from the smart contract
        """
        if not self.honestNodes:
            print("No honest nodes found!")
            return None
        
        # Initialize result bitstring with zeros
        final_bits = np.zeros(self.bitstringLength, dtype=int)
        
        # XOR all honest nodes' bitstrings to create final random number
        for node in self.honestNodes:
            final_bits = final_bits ^ self.bitstrings[node]  # Bitwise XOR operation
        
        # Store result and display
        self.finalRandomBits = final_bits
        print(f"Final random number: {''.join(map(str, final_bits))}")
        return final_bits
    
    def runFullSimulation(self) -> dict:
        """
        Run the complete QRiNG simulation
        """
        print("=" * 50)
        print("QRiNG SIMULATION STARTING")
        print("=" * 50)
        
        # Step 1: Consensus protocol
        honestNodes = self.runConsensusProtocol()
        
        # Step 2: Generate final random number
        final_random = self.generateFinalRandomNumber()
        
        print("=" * 50)
        print("SIMULATION COMPLETED")
        print("=" * 50)
        
        return {
            'honest_nodes': honestNodes,
            'final_random_number': final_random,
            'vote_counts': self.voteCounts,
            'bitstrings': self.bitstrings
        }

    # Deprecation aliases (old snake_case names still work)
    calculate_bitstring_similarity = calculateBitstringSimilarity
    perform_consensus_check = performConsensusCheck
    run_consensus_protocol = runConsensusProtocol
    generate_final_random_number = generateFinalRandomNumber
    run_full_simulation = runFullSimulation

def createQringNetworkVisualization(simulator: QRiNGSimulator, save_path: str) -> None:
    """
    Create a comprehensive network visualization of the QRiNG protocol
    """
    print("Creating QRiNG network visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 2x3 grid for different visualizations
    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Network topology with consensus results
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("QRiNG Network Topology & Consensus", fontsize=14, fontweight='bold')
    
    # Create network graph
    G = nx.complete_graph(simulator.numNodes)
    pos = nx.circular_layout(G)
    
    # Draw edges (representing QKD links)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, edge_color='lightblue')
    
    # Color nodes based on consensus results
    node_colors = []
    for node in simulator.nodes:
        # Green for honest nodes (passed consensus), yellow for suspicious, red for dishonest
        if node in simulator.honestNodes:
            node_colors.append('lightgreen')  # Honest nodes - passed consensus check
        elif simulator.voteCounts[node] > 0:
            node_colors.append('yellow')      # Suspicious nodes - some votes but not honest
        else:
            node_colors.append('lightcoral')  # Dishonest nodes - no votes received
    
    # Draw network nodes with consensus-based coloring
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)
    
    # Add node labels with vote counts for analysis
    labels = {node: f"N{node}\n({int(simulator.voteCounts[node])})" for node in simulator.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    # Create legend explaining node color coding
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=10, label='Honest Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                  markersize=10, label='Suspicious Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                  markersize=10, label='Dishonest Nodes')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 2. Bitstring comparison matrix visualization
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Bitstring Similarity Matrix", fontsize=14, fontweight='bold')
    
    # Calculate pairwise similarity between all nodes' bitstrings
    similarity_matrix = np.zeros((simulator.numNodes, simulator.numNodes))
    for i in range(simulator.numNodes):
        for j in range(simulator.numNodes):
            if i != j:
                # Use the same similarity calculation as consensus protocol
                similarity_matrix[i, j] = simulator.calculateBitstringSimilarity(i, j)
    
    # Create heatmap showing bitstring similarities (green = high similarity)
    im = ax2.imshow(similarity_matrix, cmap='RdYlGn', aspect='equal')
    ax2.set_xticks(range(simulator.numNodes))
    ax2.set_yticks(range(simulator.numNodes))
    ax2.set_xticklabels([f'N{i}' for i in range(simulator.numNodes)])
    ax2.set_yticklabels([f'N{i}' for i in range(simulator.numNodes)])
    
    # Overlay similarity values on heatmap for precise analysis
    for i in range(simulator.numNodes):
        for j in range(simulator.numNodes):
            if i != j:
                ax2.text(j, i, f'{int(similarity_matrix[i, j])}', 
                        ha='center', va='center', fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Matching Bits')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Node')
    
    # 3. Vote count distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Node Vote Count Distribution", fontsize=14, fontweight='bold')
    
    bars = ax3.bar([f'N{i}' for i in simulator.nodes], simulator.voteCounts, 
                   color=node_colors, alpha=0.7, edgecolor='black')
    
    # Add threshold line
    threshold = simulator.numNodes // 2
    ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Honesty Threshold ({threshold})')
    
    # Add value labels on bars
    for bar, count in zip(bars, simulator.voteCounts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('Vote Count')
    ax3.set_xlabel('Node')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Quantum bitstrings visualization
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Quantum Bitstrings", fontsize=14, fontweight='bold')
    
    # Create bitstring matrix for visualization
    bitstring_matrix = np.array([simulator.bitstrings[node] for node in simulator.nodes])
    
    im2 = ax4.imshow(bitstring_matrix, cmap='RdYlBu', aspect='auto')
    ax4.set_yticks(range(simulator.numNodes))
    ax4.set_yticklabels([f'N{i}' for i in simulator.nodes])
    ax4.set_xticks(range(simulator.bitstringLength))
    ax4.set_xticklabels([f'B{i}' for i in range(simulator.bitstringLength)])
    
    # Add bit values to matrix
    for i in range(simulator.numNodes):
        for j in range(simulator.bitstringLength):
            color = 'white' if bitstring_matrix[i, j] == 0 else 'black'
            ax4.text(j, i, f'{bitstring_matrix[i, j]}', 
                    ha='center', va='center', color=color, fontweight='bold')
    
    ax4.set_xlabel('Bit Position')
    ax4.set_ylabel('Node')
    
    # 5. Protocol flow diagram showing QRiNG process steps
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("QRiNG Protocol Flow", fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 4)
    ax5.axis('off')
    
    # Define the five main steps of the QRiNG protocol
    steps = [
        (1, 3, "1. QKD\nBitstring\nGeneration"),     # Step 1: Quantum key distribution
        (3, 3, "2. Consensus\nChecking"),            # Step 2: Peer validation
        (5, 3, "3. Node\nValidation"),               # Step 3: Honest node identification
        (7, 3, "4. XOR\nAggregation"),               # Step 4: Bitstring combination
        (9, 3, "5. Final\nRandom Number")            # Step 5: Result generation
    ]
    
    for i, (x, y, text) in enumerate(steps):
        # Draw colored boxes for each protocol step
        box = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor='lightblue' if i < 4 else 'lightgreen',  # Final step highlighted
                            edgecolor='black', linewidth=2)
        ax5.add_patch(box)
        
        # Add descriptive text for each step
        ax5.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw arrows showing protocol flow direction
        if i < len(steps) - 1:
            ax5.arrow(x + 0.5, y, 1, 0, head_width=0.1, head_length=0.2, 
                     fc='black', ec='black')
    
    # Display the final random number result if available
    if simulator.finalRandomBits is not None:
        final_str = ''.join(map(str, simulator.finalRandomBits))
        ax5.text(9, 2, f"Result: {final_str}", ha='center', va='center',
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Add mathematical formulas underlying the protocol
    # Bitstring similarity formula
    ax5.text(1, 1.5, r"$s_{i,j} = \{m(n) | b_i(n) = b_j(n)\}_n$", 
            fontsize=10, ha='center')
    # Matching count formula from smart contract
    ax5.text(3, 1.5, r"$match(i,j) = \sum_{n=1}^l \mathbb{I}(s_{i,j}(n) = s_{i,i}(n))$", 
            fontsize=10, ha='center')
    # Honest node threshold condition
    ax5.text(5, 1.5, r"$honest \Leftarrow match(i,j) > \frac{l}{2}$", 
            fontsize=10, ha='center')
    # Final random number generation formula
    ax5.text(7, 1.5, r"$R(n) = \bigoplus_{v_i \in V_{honest}} s_i(n)$", 
            fontsize=10, ha='center')
    
    plt.suptitle("QRiNG Protocol Simulation Results", fontsize=16, fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"QRiNG network visualization saved to {save_path}")

def createQuantumMeasurementVisualization(simulator: QRiNGSimulator, save_path: str,
                                          seed: int | None = None) -> None:
    """
    Create visualization showing quantum measurement process and Bell state correlations
    """
    print("Creating quantum measurement visualization...")
    
    rng = np.random.default_rng(seed)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bell state measurement simulation
    ax1 = axes[0, 0]
    ax1.set_title("Bell State Measurement Simulation", fontsize=14, fontweight='bold')
    
    # Simulate entangled pair measurements
    num_measurements = 1000
    alice_measurements = []
    bob_measurements = []
    
    # Generate correlated measurements (simulating |Phi+> = (|00> + |11>)/sqrt(2))
    for _ in range(num_measurements):
        if rng.random() < 0.5:
            # Both measure 0
            alice_measurements.append(0)
            bob_measurements.append(0)
        else:
            # Both measure 1
            alice_measurements.append(1)
            bob_measurements.append(1)
    
    # Add some noise to simulate real quantum systems
    noise_level = 0.05
    for i in range(len(alice_measurements)):
        if rng.random() < noise_level:
            alice_measurements[i] = 1 - alice_measurements[i]
        if rng.random() < noise_level:
            bob_measurements[i] = 1 - bob_measurements[i]
    
    # Plot correlation
    correlation_data = []
    for a, b in zip(alice_measurements[:100], bob_measurements[:100]):
        correlation_data.append([a, b])
    
    correlation_matrix = np.array(correlation_data)
    scatter = ax1.scatter(correlation_matrix[:, 0] + rng.normal(0, 0.05, len(correlation_matrix)),
                         correlation_matrix[:, 1] + rng.normal(0, 0.05, len(correlation_matrix)),
                         alpha=0.6, c=range(len(correlation_matrix)), cmap='viridis')
    
    ax1.set_xlabel("Alice's Measurement")
    ax1.set_ylabel("Bob's Measurement")
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.3, 1.3)
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display correlation coefficient
    correlation_coeff = np.corrcoef(alice_measurements, bob_measurements)[0, 1]
    ax1.text(0.05, 0.95, f"Correlation: {correlation_coeff:.3f}", 
            transform=ax1.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 2. Quantum state evolution
    ax2 = axes[0, 1]
    ax2.set_title("Quantum State Evolution", fontsize=14, fontweight='bold')
    
    # Simulate quantum state evolution during measurement
    theta_values = np.linspace(0, 2*np.pi, 100)
    prob_0 = np.cos(theta_values/2)**2
    prob_1 = np.sin(theta_values/2)**2
    
    ax2.plot(theta_values, prob_0, 'b-', label='P(|0⟩)', linewidth=2)
    ax2.plot(theta_values, prob_1, 'r-', label='P(|1⟩)', linewidth=2)
    ax2.fill_between(theta_values, prob_0, alpha=0.3, color='blue')
    ax2.fill_between(theta_values, prob_1, alpha=0.3, color='red')
    
    ax2.set_xlabel('Parameter θ')
    ax2.set_ylabel('Measurement Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(0, 1)
    
    # Add quantum state annotations
    ax2.text(np.pi/4, 0.8, r'$|\psi\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$', 
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # 3. Randomness quality analysis
    ax3 = axes[1, 0]
    ax3.set_title("Randomness Quality Analysis", fontsize=14, fontweight='bold')
    
    # Analyze randomness of generated bitstrings
    all_bits = np.concatenate([simulator.bitstrings[node] for node in simulator.nodes])
    
    # Calculate entropy
    p_0 = np.sum(all_bits == 0) / len(all_bits)
    p_1 = np.sum(all_bits == 1) / len(all_bits)
    entropy = -p_0 * np.log2(p_0 + 1e-10) - p_1 * np.log2(p_1 + 1e-10)
    
    # Frequency analysis
    frequencies = [np.mean(simulator.bitstrings[node]) for node in simulator.nodes]
    
    bars = ax3.bar([f'N{i}' for i in simulator.nodes], frequencies, 
                   color='lightblue', alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect Randomness')
    
    ax3.set_ylabel('Bit Frequency (P(1))')
    ax3.set_xlabel('Node')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add entropy information
    ax3.text(0.05, 0.95, f"Total Entropy: {entropy:.3f} bits\nMax Entropy: 1.000 bits", 
            transform=ax3.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 4. XOR operation visualization
    ax4 = axes[1, 1]
    ax4.set_title("XOR Aggregation Process", fontsize=14, fontweight='bold')
    
    if simulator.honestNodes and len(simulator.honestNodes) > 1:
        # Show step-by-step XOR process
        y_pos = len(simulator.honestNodes)
        colors = plt.cm.Set3(np.linspace(0, 1, len(simulator.honestNodes)))
        
        cumulative_xor = np.zeros(simulator.bitstringLength, dtype=int)
        
        for i, node in enumerate(simulator.honestNodes):
            bitstring = simulator.bitstrings[node]
            
            # Display individual bitstring
            for j, bit in enumerate(bitstring):
                color = 'white' if bit == 0 else colors[i]
                rect = patches.Rectangle((j, y_pos - i - 1), 1, 0.8, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor=color, alpha=0.8)
                ax4.add_patch(rect)
                ax4.text(j + 0.5, y_pos - i - 0.5, str(bit), 
                        ha='center', va='center', fontweight='bold')
            
            # Update cumulative XOR
            cumulative_xor = cumulative_xor ^ bitstring
            
            # Add node label
            ax4.text(-0.5, y_pos - i - 0.5, f'N{node}', 
                    ha='center', va='center', fontweight='bold')
        
        # Show final result
        ax4.text(-0.5, -1.5, 'Result', ha='center', va='center', fontweight='bold', fontsize=12)
        for j, bit in enumerate(cumulative_xor):
            color = 'lightcoral' if bit == 0 else 'lightgreen'
            rect = patches.Rectangle((j, -2), 1, 0.8, 
                                   linewidth=2, edgecolor='black', 
                                   facecolor=color, alpha=0.9)
            ax4.add_patch(rect)
            ax4.text(j + 0.5, -1.5, str(bit), 
                    ha='center', va='center', fontweight='bold', fontsize=12)
        
        ax4.set_xlim(-1, simulator.bitstringLength)
        ax4.set_ylim(-3, len(simulator.honestNodes))
        ax4.set_xlabel('Bit Position')
        ax4.set_ylabel('Node / Result')
        
        # Add XOR symbols
        for i in range(len(simulator.honestNodes) - 1):
            ax4.text(simulator.bitstringLength + 0.5, y_pos - i - 1.5, '⊕', 
                    ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Quantum measurement visualization saved to {save_path}")

def runSimulationAndCreateVisualizations() -> tuple[QRiNGSimulator, dict]:
    """
    Run the complete QRiNG simulation and generate all visualizations
    """
    print("Starting QRiNG Simulation and Visualization Generation")
    print("=" * 60)
    
    # Initialize simulator with reproducible seed
    simulator = QRiNGSimulator(numNodes=6, bitstringLength=8, seed=42)
    
    # Run the simulation
    results = simulator.runFullSimulation()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")
    
    # Generate visualizations
    network_viz_path = os.path.join(output_dir, "qring_simulator_network.png")
    quantum_viz_path = os.path.join(output_dir, "qring_simulator_quantum.png")
    
    createQringNetworkVisualization(simulator, network_viz_path)
    createQuantumMeasurementVisualization(simulator, quantum_viz_path)
    
    # Print comprehensive simulation summary for analysis
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total nodes: {simulator.numNodes}")
    print(f"Bitstring length: {simulator.bitstringLength}")
    print(f"Honest nodes: {len(simulator.honestNodes)} ({simulator.honestNodes})")
    print(f"Final random number: {''.join(map(str, simulator.finalRandomBits)) if simulator.finalRandomBits is not None else 'None'}")
    print(f"Vote counts: {dict(zip(simulator.nodes, simulator.voteCounts))}")
    
    # List generated visualization files for user reference
    print("\nVisualization files created:")
    print(f"- Network visualization: {network_viz_path}")
    print(f"- Quantum measurement visualization: {quantum_viz_path}")
    
    return simulator, results

# Deprecation aliases for standalone functions
create_qring_network_visualization = createQringNetworkVisualization
create_quantum_measurement_visualization = createQuantumMeasurementVisualization
run_simulation_and_create_visualizations = runSimulationAndCreateVisualizations

if __name__ == "__main__":
    try:
        # Execute main simulation and visualization generation
        simulator, results = runSimulationAndCreateVisualizations()
        print("\nQRiNG simulation completed successfully!")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
