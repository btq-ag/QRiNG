"""
QRiNG Simulator - Quantum Random Number Generator Simulation

This script simulates the QRiNG protocol by modeling:
1. Quantum Key Distribution (QKD) between network nodes
2. Consensus mechanism for node validation
3. Final random number generation through XOR aggregation
4. Comprehensive visualizations of the entire process

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import seaborn as sns
import os
from datetime import datetime

class QRiNGSimulator:
    """
    Simulator for the QRiNG (Quantum Random Number Generator) protocol
    """
    
    def __init__(self, num_nodes=6, bitstring_length=8, seed=None):
        """
        Initialize the QRiNG simulator
        
        Args:
            num_nodes (int): Number of nodes in the network
            bitstring_length (int): Length of quantum bitstrings
            seed (int): Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.num_nodes = num_nodes
        self.bitstring_length = bitstring_length
        self.nodes = list(range(num_nodes))
        self.bitstrings = {}
        self.vote_counts = np.zeros(num_nodes)
        self.has_voted = np.zeros(num_nodes, dtype=bool)
        self.honest_nodes = []
        self.final_random_bits = None
        
        # Generate quantum bitstrings through simulated QKD
        self._generate_quantum_bitstrings()
        
    def _generate_quantum_bitstrings(self):
        """
        Simulate quantum key distribution to generate bitstrings for each node
        Uses Bell state measurements and quantum entanglement simulation
        """
        print("Generating quantum bitstrings through simulated QKD...")
        
        for node in self.nodes:
            # Simulate quantum measurement outcomes
            # Each bit has quantum uncertainty with bias towards true randomness
            quantum_bias = 0.5 + np.random.normal(0, 0.05)  # Slight deviation from perfect 50/50
            quantum_bias = np.clip(quantum_bias, 0.3, 0.7)  # Keep within reasonable bounds
              # Generate bitstring with quantum-like properties
            bitstring = []
            for bit_idx in range(self.bitstring_length):
                # Simulate quantum measurement with entanglement correlations
                if bit_idx > 0:
                    # Add correlation with previous bit (simulating entanglement)
                    correlation_factor = 0.15 * np.sin(bit_idx * np.pi / 4)
                    prob = quantum_bias + correlation_factor * (2 * bitstring[-1] - 1)
                    prob = np.clip(prob, 0.1, 0.9)  # Ensure valid probability range
                else:
                    prob = quantum_bias
                
                # Generate quantum bit based on calculated probability
                bit = 1 if np.random.random() < prob else 0
                bitstring.append(bit)
            
            self.bitstrings[node] = np.array(bitstring)
            
        print(f"Generated {len(self.bitstrings)} quantum bitstrings")
    
    def calculate_bitstring_similarity(self, node1, node2):
        """
        Calculate similarity between two nodes' bitstrings
        Implements the matching function from the smart contract
        """
        if node1 not in self.bitstrings or node2 not in self.bitstrings:
            return 0
        
        matches = np.sum(self.bitstrings[node1] == self.bitstrings[node2])
        return matches
    
    def perform_consensus_check(self, checking_node):
        """
        Simulate the consensus check function from the smart contract
        Each node checks all other nodes' bitstrings
        """
        if self.has_voted[checking_node]:
            return False
        
        threshold = self.bitstring_length // 2
        
        for target_node in self.nodes:
            if target_node != checking_node:
                # Calculate similarity between bitstrings
                matches = self.calculate_bitstring_similarity(checking_node, target_node)
                # If similarity exceeds threshold, increment vote count
                if matches > threshold:
                    self.vote_counts[target_node] += 1
        
        # Mark this node as having voted to prevent double-voting
        self.has_voted[checking_node] = True
        return True
    
    def run_consensus_protocol(self):
        """
        Run the complete consensus protocol with all nodes participating
        """
        print("Running consensus protocol...")
        
        # Each node performs checking
        for node in self.nodes:
            self.perform_consensus_check(node)
          # Determine honest nodes (those with vote count > num_nodes/2)
        threshold = len(self.nodes) // 2
        self.honest_nodes = [node for node in self.nodes if self.vote_counts[node] > threshold]
        
        print(f"Honest nodes identified: {self.honest_nodes}")
        return self.honest_nodes
    
    def generate_final_random_number(self):
        """
        Generate final random number by XOR-ing honest nodes' bitstrings
        Implements the randomNumber() function from the smart contract
        """
        if not self.honest_nodes:
            print("No honest nodes found!")
            return None
        
        # Initialize result bitstring with zeros
        final_bits = np.zeros(self.bitstring_length, dtype=int)
        
        # XOR all honest nodes' bitstrings to create final random number
        for node in self.honest_nodes:
            final_bits = final_bits ^ self.bitstrings[node]  # Bitwise XOR operation
        
        # Store result and display
        self.final_random_bits = final_bits
        print(f"Final random number: {''.join(map(str, final_bits))}")
        return final_bits
    
    def run_full_simulation(self):
        """
        Run the complete QRiNG simulation
        """
        print("=" * 50)
        print("QRiNG SIMULATION STARTING")
        print("=" * 50)
        
        # Step 1: Consensus protocol
        honest_nodes = self.run_consensus_protocol()
        
        # Step 2: Generate final random number
        final_random = self.generate_final_random_number()
        
        print("=" * 50)
        print("SIMULATION COMPLETED")
        print("=" * 50)
        
        return {
            'honest_nodes': honest_nodes,
            'final_random': final_random,
            'vote_counts': self.vote_counts,
            'bitstrings': self.bitstrings
        }

def create_qring_network_visualization(simulator, save_path):
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
    G = nx.complete_graph(simulator.num_nodes)
    pos = nx.circular_layout(G)
    
    # Draw edges (representing QKD links)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, edge_color='lightblue')
    
    # Color nodes based on consensus results
    node_colors = []
    for node in simulator.nodes:
        # Green for honest nodes (passed consensus), yellow for suspicious, red for dishonest
        if node in simulator.honest_nodes:
            node_colors.append('lightgreen')  # Honest nodes - passed consensus check
        elif simulator.vote_counts[node] > 0:
            node_colors.append('yellow')      # Suspicious nodes - some votes but not honest
        else:
            node_colors.append('lightcoral')  # Dishonest nodes - no votes received
    
    # Draw network nodes with consensus-based coloring
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)
    
    # Add node labels with vote counts for analysis
    labels = {node: f"N{node}\n({int(simulator.vote_counts[node])})" for node in simulator.nodes}
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
    similarity_matrix = np.zeros((simulator.num_nodes, simulator.num_nodes))
    for i in range(simulator.num_nodes):
        for j in range(simulator.num_nodes):
            if i != j:
                # Use the same similarity calculation as consensus protocol
                similarity_matrix[i, j] = simulator.calculate_bitstring_similarity(i, j)
    
    # Create heatmap showing bitstring similarities (green = high similarity)
    im = ax2.imshow(similarity_matrix, cmap='RdYlGn', aspect='equal')
    ax2.set_xticks(range(simulator.num_nodes))
    ax2.set_yticks(range(simulator.num_nodes))
    ax2.set_xticklabels([f'N{i}' for i in range(simulator.num_nodes)])
    ax2.set_yticklabels([f'N{i}' for i in range(simulator.num_nodes)])
    
    # Overlay similarity values on heatmap for precise analysis
    for i in range(simulator.num_nodes):
        for j in range(simulator.num_nodes):
            if i != j:
                ax2.text(j, i, f'{int(similarity_matrix[i, j])}', 
                        ha='center', va='center', fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Matching Bits')
    ax2.set_xlabel('Node')
    ax2.set_ylabel('Node')
    
    # 3. Vote count distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Node Vote Count Distribution", fontsize=14, fontweight='bold')
    
    bars = ax3.bar([f'N{i}' for i in simulator.nodes], simulator.vote_counts, 
                   color=node_colors, alpha=0.7, edgecolor='black')
    
    # Add threshold line
    threshold = simulator.num_nodes // 2
    ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Honesty Threshold ({threshold})')
    
    # Add value labels on bars
    for bar, count in zip(bars, simulator.vote_counts):
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
    ax4.set_yticks(range(simulator.num_nodes))
    ax4.set_yticklabels([f'N{i}' for i in simulator.nodes])
    ax4.set_xticks(range(simulator.bitstring_length))
    ax4.set_xticklabels([f'B{i}' for i in range(simulator.bitstring_length)])
    
    # Add bit values to matrix
    for i in range(simulator.num_nodes):
        for j in range(simulator.bitstring_length):
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
    if simulator.final_random_bits is not None:
        final_str = ''.join(map(str, simulator.final_random_bits))
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

def create_quantum_measurement_visualization(simulator, save_path):
    """
    Create visualization showing quantum measurement process and Bell state correlations
    """
    print("Creating quantum measurement visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bell state measurement simulation
    ax1 = axes[0, 0]
    ax1.set_title("Bell State Measurement Simulation", fontsize=14, fontweight='bold')
    
    # Simulate entangled pair measurements
    num_measurements = 1000
    alice_measurements = []
    bob_measurements = []
    
    # Generate correlated measurements (simulating |Φ+⟩ = (|00⟩ + |11⟩)/√2)
    for _ in range(num_measurements):
        if np.random.random() < 0.5:
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
        if np.random.random() < noise_level:
            alice_measurements[i] = 1 - alice_measurements[i]
        if np.random.random() < noise_level:
            bob_measurements[i] = 1 - bob_measurements[i]
    
    # Plot correlation
    correlation_data = []
    for a, b in zip(alice_measurements[:100], bob_measurements[:100]):
        correlation_data.append([a, b])
    
    correlation_matrix = np.array(correlation_data)
    scatter = ax1.scatter(correlation_matrix[:, 0] + np.random.normal(0, 0.05, len(correlation_matrix)),
                         correlation_matrix[:, 1] + np.random.normal(0, 0.05, len(correlation_matrix)),
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
    
    if simulator.honest_nodes and len(simulator.honest_nodes) > 1:
        # Show step-by-step XOR process
        y_pos = len(simulator.honest_nodes)
        colors = plt.cm.Set3(np.linspace(0, 1, len(simulator.honest_nodes)))
        
        cumulative_xor = np.zeros(simulator.bitstring_length, dtype=int)
        
        for i, node in enumerate(simulator.honest_nodes):
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
        
        ax4.set_xlim(-1, simulator.bitstring_length)
        ax4.set_ylim(-3, len(simulator.honest_nodes))
        ax4.set_xlabel('Bit Position')
        ax4.set_ylabel('Node / Result')
        
        # Add XOR symbols
        for i in range(len(simulator.honest_nodes) - 1):
            ax4.text(simulator.bitstring_length + 0.5, y_pos - i - 1.5, '⊕', 
                    ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Quantum measurement visualization saved to {save_path}")

def run_simulation_and_create_visualizations():
    """
    Run the complete QRiNG simulation and generate all visualizations
    """
    print("Starting QRiNG Simulation and Visualization Generation")
    print("=" * 60)
    
    # Initialize simulator with reproducible seed
    simulator = QRiNGSimulator(num_nodes=6, bitstring_length=8, seed=42)
    
    # Run the simulation
    results = simulator.run_full_simulation()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")
    
    # Generate visualizations
    network_viz_path = os.path.join(output_dir, "qring_simulator_network.png")
    quantum_viz_path = os.path.join(output_dir, "qring_simulator_quantum.png")
    
    create_qring_network_visualization(simulator, network_viz_path)
    create_quantum_measurement_visualization(simulator, quantum_viz_path)
    
    # Print comprehensive simulation summary for analysis
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total nodes: {simulator.num_nodes}")
    print(f"Bitstring length: {simulator.bitstring_length}")
    print(f"Honest nodes: {len(simulator.honest_nodes)} ({simulator.honest_nodes})")
    print(f"Final random number: {''.join(map(str, simulator.final_random_bits)) if simulator.final_random_bits is not None else 'None'}")
    print(f"Vote counts: {dict(zip(simulator.nodes, simulator.vote_counts))}")
    
    # List generated visualization files for user reference
    print("\nVisualization files created:")
    print(f"- Network visualization: {network_viz_path}")
    print(f"- Quantum measurement visualization: {quantum_viz_path}")
    
    return simulator, results

if __name__ == "__main__":
    try:
        # Execute main simulation and visualization generation
        simulator, results = run_simulation_and_create_visualizations()
        print("\nQRiNG simulation completed successfully!")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
