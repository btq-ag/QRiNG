"""
QRiNG Emulator - Smart Contract Function Emulation

This script emulates the exact behavior of the QRiNG smart contract functions:
1. Replicates the Solidity contract logic in Python
2. Provides detailed step-by-step execution visualization
3. Demonstrates gas consumption and blockchain interaction patterns
4. Creates comprehensive visualizations of the emulated smart contract execution

Author: Jeffrey Morais, BTQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns
import os
from datetime import datetime
import json

class QRiNGEmulator:
    """
    Emulator for the QRiNG smart contract that replicates exact Solidity behavior
    """
    
    def __init__(self, bitstring_length=6):
        """
        Initialize the QRiNG emulator to match smart contract structure
        
        Args:
            bitstring_length (int): Length of bitstrings (matches smart contract array size)
        """
        # Smart contract state variables
        self.voters = []  # Array of Voter structs
        self.admin = None
        self.voting_active = False
        self.counter = []  # 2D array for bitstrings
        self.bitstring_length = bitstring_length
        
        # Emulation tracking
        self.transaction_log = []
        self.gas_consumption = {}
        self.events_emitted = []
        
        # Address simulation (using string addresses like Ethereum)
        self.addresses = []
    
    def _log_transaction(self, function_name, caller, gas_used, success=True, revert_reason=None):
        """Log transaction details for emulation tracking"""
        tx = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'caller': caller,
            'gas_used': gas_used,
            'success': success,
            'revert_reason': revert_reason,
            'block_number': len(self.transaction_log) + 1
        }
        self.transaction_log.append(tx)
        
        if function_name not in self.gas_consumption:
            self.gas_consumption[function_name] = []
        self.gas_consumption[function_name].append(gas_used)
    
    def _emit_event(self, event_name, event_data):
        """Emulate event emission"""
        event = {
            'event': event_name,
            'data': event_data,
            'block_number': len(self.transaction_log) + 1,
            'timestamp': datetime.now().isoformat()
        }
        self.events_emitted.append(event)
    
    def add_new_string(self, new_string, caller_address):
        """
        Emulate addNewString function from smart contract
        function addNewString(uint[][] memory newString) public
        """
        print(f"Executing addNewString() called by {caller_address}")
        
        # Gas calculation (approximate)
        gas_used = 21000 + len(new_string) * len(new_string[0]) * 20  # Base + storage cost
        
        try:
            # Store the new bitstrings
            self.counter = [list(bitstring) for bitstring in new_string]
            
            self._log_transaction('addNewString', caller_address, gas_used, True)
            print(f"✓ Bitstrings stored successfully. Gas used: {gas_used}")
            return True
            
        except Exception as e:
            self._log_transaction('addNewString', caller_address, gas_used, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def set_addresses(self, voter_addresses, caller_address):
        """
        Emulate setAddresses function from smart contract
        function setAddresses(address[] memory voterAddresses) public
        """
        print(f"Executing setAddresses() called by {caller_address}")
        
        # Gas calculation based on number of voters to register
        gas_used = 21000 + len(voter_addresses) * 50000  # Base + voter creation cost
        
        try:
            # Initialize admin and voting system state
            self.admin = caller_address
            self.voting_active = True
            self.addresses = voter_addresses
            
            # Create voter structs matching Solidity contract structure
            self.voters = []
            for x, address in enumerate(voter_addresses):
                if x < len(self.counter):
                    # Create voter struct matching Solidity contract fields
                    voter = {
                        'delegate': address,         # Voter's Ethereum address
                        'hasVoted': False,          # Voting participation flag
                        'number': x,                # Voter index/ID
                        'voteCount': 0,             # Number of votes received
                        'bitstring': self.counter[x].copy()  # Pre-assigned quantum bitstring
                    }
                    self.voters.append(voter)
                    
                    # Emit event for off-chain monitoring
                    self._emit_event('VoterRegistered', {'voter': address})
            
            self._log_transaction('setAddresses', caller_address, gas_used, True)
            print(f"✓ {len(self.voters)} voters registered. Admin set to {self.admin}")
            print(f"  Gas used: {gas_used}")
            return True
            
        except Exception as e:
            self._log_transaction('setAddresses', caller_address, gas_used, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def start_voting(self, caller_address):
        """
        Emulate startVoting function from smart contract
        function startVoting() external onlyAdmin
        """
        print(f"Executing startVoting() called by {caller_address}")
        
        gas_used = 21000  # Base gas cost
        
        try:
            # Enforce admin-only access control (onlyAdmin modifier)
            if caller_address != self.admin:
                raise Exception("Only admin can call this function")
            
            # Prevent duplicate voting activation
            if self.voting_active:
                raise Exception("Voting is already active")
            
            # Activate the voting process
            self.voting_active = True
            
            self._log_transaction('startVoting', caller_address, gas_used, True)
            print(f"✓ Voting started. Gas used: {gas_used}")
            return True
            
        except Exception as e:
            self._log_transaction('startVoting', caller_address, gas_used, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def end_voting(self, caller_address):
        """
        Emulate endVoting function from smart contract
        function endVoting() external onlyAdmin
        """
        print(f"Executing endVoting() called by {caller_address}")
        
        gas_used = 21000  # Base gas cost
        
        try:
            # Enforce admin-only access for ending voting
            if caller_address != self.admin:
                raise Exception("Only admin can call this function")
            
            # Ensure voting is currently active before ending
            if not self.voting_active:
                raise Exception("Voting is not active")
            
            # Deactivate voting system
            self.voting_active = False
            
            # Notify external systems that voting has concluded
            self._emit_event('VotingEnded', {})
            
            self._log_transaction('endVoting', caller_address, gas_used, True)
            print(f"✓ Voting ended. Gas used: {gas_used}")
            return True
            
        except Exception as e:
            self._log_transaction('endVoting', caller_address, gas_used, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def check(self, from_node, caller_address):
        """
        Emulate check function from smart contract
        function check(uint from) external onlyActive
        """
        print(f"Executing check({from_node}) called by {caller_address}")
        
        # Gas calculation (complex function with loops)
        gas_used = 21000 + len(self.voters) * len(self.voters[0]['bitstring']) * 10
        
        try:
            # Check onlyActive modifier
            if not self.voting_active:
                raise Exception("Voting is not active")
            
            # Check authorization
            if from_node >= len(self.voters) or self.voters[from_node]['delegate'] != caller_address:
                raise Exception("Not authorized")
            
            # Check if already voted
            if self.voters[from_node]['hasVoted']:
                raise Exception("Voter has already voted")
              # Perform the check logic (exact replication of Solidity code)
            bitstring_length = len(self.voters[from_node]['bitstring']) // 2  # Threshold for correlation
            
            # Compare this voter's bitstring with all other voters
            for i in range(len(self.voters)):
                if from_node != i:
                    add = 0  # Counter for matching bits
                    # Count matching bits between bitstrings
                    for j in range(len(self.voters[from_node]['bitstring'])):
                        if (self.voters[from_node]['bitstring'][j] == 
                            self.voters[i]['bitstring'][j]):
                            add += 1
                    
                    # If correlation exceeds threshold, increment vote count
                    if add > bitstring_length:
                        self.voters[i]['voteCount'] += 1
            
            # Mark as voted to prevent double-voting
            self.voters[from_node]['hasVoted'] = True
            
            self._log_transaction('check', caller_address, gas_used, True)
            print(f"✓ Check completed for node {from_node}. Gas used: {gas_used}")
            return True
            
        except Exception as e:
            self._log_transaction('check', caller_address, gas_used, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def get_winner(self, caller_address):
        """
        Emulate getWinner function from smart contract
        function getWinner() external view returns (uint8)
        """
        print(f"Executing getWinner() called by {caller_address}")
        
        gas_used = 5000  # View function, lower gas cost
        
        try:
            # Check if voting is still active
            if self.voting_active:
                raise Exception("Voting is still active")
            
            sum_count = 0
            length = len(self.voters) // 2
            
            for i in range(len(self.voters)):
                if self.voters[i]['voteCount'] > length:
                    sum_count += 1
            
            self._log_transaction('getWinner', caller_address, gas_used, True)
            print(f"✓ getWinner() returned {sum_count}. Gas used: {gas_used}")
            return sum_count
            
        except Exception as e:
            self._log_transaction('getWinner', caller_address, gas_used, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return None
    
    def random_number(self, caller_address):
        """
        Emulate randomNumber function from smart contract
        function randomNumber() external view returns (uint[6] memory newBitstring)
        """
        print(f"Executing randomNumber() called by {caller_address}")
        
        gas_used = 10000  # View function with computation
        
        try:            # Check if voting is still active
            if self.voting_active:
                raise Exception("Voting is still active")
            
            # Initialize result bitstring with zeros
            new_bitstring = [0] * self.bitstring_length
            length = len(self.voters) // 2  # Majority threshold
            
            # XOR operation on honest nodes (exact Solidity replication)
            for i in range(len(self.voters)):
                # Only include voters with sufficient vote count (honest nodes)
                if self.voters[i]['voteCount'] > length:
                    # XOR each bit position
                    for x in range(len(self.voters[i]['bitstring'])):
                        if x < self.bitstring_length:  # Ensure we don't exceed array bounds
                            new_bitstring[x] += self.voters[i]['bitstring'][x]
                            new_bitstring[x] %= 2  # Ensure binary result (XOR operation)
            
            self._log_transaction('randomNumber', caller_address, gas_used, True)
            print(f"✓ randomNumber() returned {new_bitstring}. Gas used: {gas_used}")
            return new_bitstring
            
        except Exception as e:
            self._log_transaction('randomNumber', caller_address, gas_used, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return None
    
    def get_voter_info(self, voter_index):
        """Helper function to get voter information"""
        if 0 <= voter_index < len(self.voters):
            return self.voters[voter_index].copy()
        return None
    
    def get_contract_state(self):
        """Get complete contract state for visualization"""
        return {
            'admin': self.admin,
            'voting_active': self.voting_active,
            'num_voters': len(self.voters),
            'voters': [voter.copy() for voter in self.voters],
            'transaction_count': len(self.transaction_log),
            'events_count': len(self.events_emitted)
        }

def create_smart_contract_execution_visualization(emulator, save_path):
    """
    Create comprehensive visualization of smart contract execution
    """
    print("Creating smart contract execution visualization...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Transaction timeline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Smart Contract Transaction Timeline", fontsize=16, fontweight='bold')
    
    if emulator.transaction_log:
        timestamps = [i for i in range(len(emulator.transaction_log))]
        functions = [tx['function'] for tx in emulator.transaction_log]
        gas_used = [tx['gas_used'] for tx in emulator.transaction_log]
        success = [tx['success'] for tx in emulator.transaction_log]
        
        # Color code by success/failure
        colors = ['green' if s else 'red' for s in success]
        
        bars = ax1.bar(timestamps, gas_used, color=colors, alpha=0.7, edgecolor='black')
        
        # Add function labels
        for i, (bar, func) in enumerate(zip(bars, functions)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(gas_used) * 0.01,
                    func, ha='center', va='bottom', rotation=45, fontsize=9)
        
        ax1.set_xlabel('Transaction Number')
        ax1.set_ylabel('Gas Used')
        ax1.set_title('Transaction Timeline with Gas Consumption')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=4, label='Successful'),
            Line2D([0], [0], color='red', lw=4, label='Failed')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. Voter state matrix
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Voter State Matrix", fontsize=14, fontweight='bold')
    
    if emulator.voters:
        # Create state matrix
        state_data = []
        voter_labels = []
        
        for i, voter in enumerate(emulator.voters):
            voter_labels.append(f"V{i}")
            row = [
                voter['voteCount'],
                1 if voter['hasVoted'] else 0,
                voter['number']
            ]
            state_data.append(row)
        
        state_matrix = np.array(state_data)
        
        im = ax2.imshow(state_matrix, cmap='RdYlGn', aspect='auto')
        ax2.set_yticks(range(len(voter_labels)))
        ax2.set_yticklabels(voter_labels)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Vote Count', 'Has Voted', 'Number'])
        
        # Add values to cells
        for i in range(len(voter_labels)):
            for j in range(3):
                ax2.text(j, i, f'{state_matrix[i, j]}', 
                        ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='Value')
    
    # 3. Bitstring comparison heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Bitstring Similarity Heatmap", fontsize=14, fontweight='bold')
    
    if emulator.voters:
        num_voters = len(emulator.voters)
        similarity_matrix = np.zeros((num_voters, num_voters))
        
        for i in range(num_voters):
            for j in range(num_voters):
                if i != j:
                    matches = sum(1 for a, b in zip(emulator.voters[i]['bitstring'], 
                                                  emulator.voters[j]['bitstring']) if a == b)
                    similarity_matrix[i, j] = matches
        
        im3 = ax3.imshow(similarity_matrix, cmap='Blues', aspect='equal')
        ax3.set_xticks(range(num_voters))
        ax3.set_yticks(range(num_voters))
        ax3.set_xticklabels([f'V{i}' for i in range(num_voters)])
        ax3.set_yticklabels([f'V{i}' for i in range(num_voters)])
        
        # Add similarity values
        for i in range(num_voters):
            for j in range(num_voters):
                if i != j:
                    ax3.text(j, i, f'{int(similarity_matrix[i, j])}', 
                            ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im3, ax=ax3, label='Matching Bits')
    
    # 4. Gas consumption analysis
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("Gas Consumption by Function", fontsize=14, fontweight='bold')
    
    if emulator.gas_consumption:
        functions = list(emulator.gas_consumption.keys())
        avg_gas = [np.mean(emulator.gas_consumption[func]) for func in functions]
        
        bars = ax4.bar(functions, avg_gas, color='lightblue', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, gas in zip(bars, avg_gas):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(avg_gas) * 0.01,
                    f'{int(gas)}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Average Gas Used')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    # 5. Contract state flow diagram
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("Smart Contract State Flow", fontsize=16, fontweight='bold')
    ax5.set_xlim(0, 12)
    ax5.set_ylim(0, 6)
    ax5.axis('off')
    
    # Draw contract lifecycle
    states = [
        (1, 4, "Contract\nDeployed", 'lightgray'),
        (3, 4, "Bitstrings\nAdded", 'lightblue'),
        (5, 4, "Addresses\nSet", 'lightgreen'),
        (7, 4, "Voting\nActive", 'yellow'),
        (9, 4, "Checks\nCompleted", 'orange'),
        (11, 4, "Random Number\nGenerated", 'lightcoral')
    ]
    
    for i, (x, y, text, color) in enumerate(states):
        # Draw state box
        box = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax5.add_patch(box)
        
        # Add state text
        ax5.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw arrow to next state
        if i < len(states) - 1:
            ax5.arrow(x + 0.7, y, 1.6, 0, head_width=0.2, head_length=0.3, 
                     fc='black', ec='black', alpha=0.7)
    
    # Add function calls below
    functions_timeline = [
        (2, 2.5, "addNewString()"),
        (4, 2.5, "setAddresses()"),
        (6, 2.5, "check() × N"),
        (8, 2.5, "endVoting()"),
        (10, 2.5, "randomNumber()")
    ]
    
    for x, y, func_text in functions_timeline:
        ax5.text(x, y, func_text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 6. Detailed bitstring visualization
    ax6 = fig.add_subplot(gs[3, :2])
    ax6.set_title("Bitstring Details", fontsize=14, fontweight='bold')
    
    if emulator.voters:
        num_voters = len(emulator.voters)
        bitstring_length = len(emulator.voters[0]['bitstring'])
        
        # Create bitstring matrix
        bitstring_matrix = np.array([voter['bitstring'] for voter in emulator.voters])
        
        im6 = ax6.imshow(bitstring_matrix, cmap='RdYlBu', aspect='auto')
        ax6.set_yticks(range(num_voters))
        ax6.set_yticklabels([f"V{i} (VC:{emulator.voters[i]['voteCount']})" 
                            for i in range(num_voters)])
        ax6.set_xticks(range(bitstring_length))
        ax6.set_xticklabels([f'B{i}' for i in range(bitstring_length)])
        
        # Add bit values
        for i in range(num_voters):
            for j in range(bitstring_length):
                color = 'white' if bitstring_matrix[i, j] == 0 else 'black'
                ax6.text(j, i, f'{bitstring_matrix[i, j]}', 
                        ha='center', va='center', color=color, fontweight='bold')
        
        ax6.set_xlabel('Bit Position')
        ax6.set_ylabel('Voter (Vote Count)')
      # 7. Events and logs
    ax7 = fig.add_subplot(gs[3, 2])
    ax7.set_title("Contract Events", fontsize=14, fontweight='bold')
    ax7.axis('off')
    
    if emulator.events_emitted:
        event_text = "Emitted Events:\n\n"
        for i, event in enumerate(emulator.events_emitted[-5:]):  # Show last 5 events
            event_text += f"{i+1}. {event['event']}\n"
            if 'voter' in event['data']:
                event_text += f"   Voter: {event['data']['voter'][:10]}...\n"
            event_text += f"   Block: {event['block_number']}\n\n"
    else:
        event_text = "No events emitted yet"
    
    ax7.text(0.05, 0.95, event_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle("QRiNG Smart Contract Emulation Results", fontsize=18, fontweight='bold')
      # Save the figure with proper spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.93, bottom=0.07, left=0.08, right=0.95)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Smart contract execution visualization saved to {save_path}")

def run_emulation_demo():
    """
    Run a complete demonstration of the QRiNG smart contract emulation
    """
    print("Starting QRiNG Smart Contract Emulation Demo")
    print("=" * 60)
    
    # Initialize emulator
    emulator = QRiNGEmulator(bitstring_length=6)
    
    # Generate test data (6 nodes with 6-bit strings to match smart contract)
    test_bitstrings = [
        [1, 0, 1, 1, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1]
    ]
    
    # Generate test addresses
    test_addresses = [
        "0x742d35Cc6473C4C6D7d4b7d6F1d7C8A4B9E2F3A1",
        "0x8B3A4C9D2E1F5G7H8I9J0K1L2M3N4O5P6Q7R8S9T",
        "0x9F2E8D7C6B5A4938271605C4B3A2F1E0D9C8B7A6",
        "0xA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0",
        "0xB5F4E3D2C1B0A9F8E7D6C5B4A3F2E1D0C9B8A7F6",
        "0xC8A7F6E5D4C3B2A1F0E9D8C7B6A5F4E3D2C1B0A9"
    ]
    
    admin_address = "0x1234567890123456789012345678901234567890"
    
    print("Phase 1: Contract Deployment and Setup")
    print("-" * 40)
    
    # 1. Add bitstrings
    emulator.add_new_string(test_bitstrings, admin_address)
    
    # 2. Set addresses
    emulator.set_addresses(test_addresses, admin_address)
    
    print("\nPhase 2: Voting Process")
    print("-" * 40)
    
    # 3. Each voter performs check
    for i, address in enumerate(test_addresses):
        emulator.check(i, address)
    
    print("\nPhase 3: Results Generation")
    print("-" * 40)
    
    # 4. End voting
    emulator.end_voting(admin_address)
      # 5. Get results
    winner_count = emulator.get_winner(admin_address)
    final_random = emulator.random_number(admin_address)
    
    print(f"\nEMULATION RESULTS:")
    print(f"Honest nodes count: {winner_count}")
    print(f"Final random number: {final_random}")
    
    # Create visualizations
    output_dir = r"c:\Users\hunkb\OneDrive\Desktop\btq\QRiNG\Plots"
    execution_viz_path = os.path.join(output_dir, "qring_emulator_execution.png")
    
    create_smart_contract_execution_visualization(emulator, execution_viz_path)
    
    print(f"\nVisualization files created:")
    print(f"- Execution visualization: {execution_viz_path}")
    
    return emulator

if __name__ == "__main__":
    try:
        # Execute the complete smart contract emulation demonstration
        emulator = run_emulation_demo()
        print("\nQRiNG smart contract emulation completed successfully!")
        
        # Display final contract state for verification and analysis
        state = emulator.get_contract_state()
        print(f"\nFinal Contract State:")
        print(f"- Admin: {state['admin'][:10]}...")      # Show first 10 chars of admin address
        print(f"- Voting Active: {state['voting_active']}")
        print(f"- Number of Voters: {state['num_voters']}")
        print(f"- Total Transactions: {state['transaction_count']}")
        print(f"- Events Emitted: {state['events_count']}")
        
    except Exception as e:
        # Handle any errors during emulation execution
        print(f"Error during emulation: {e}")
        import traceback
        traceback.print_exc()
