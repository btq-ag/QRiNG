"""
QRiNG Emulator - Smart Contract Function Emulation

This script emulates the exact behavior of the QRiNG smart contract functions:
1. Replicates the Solidity contract logic in Python
2. Provides detailed step-by-step execution visualization
3. Demonstrates gas consumption and blockchain interaction patterns
4. Creates comprehensive visualizations of the emulated smart contract execution

Author: Jeffrey Morais, BTQ
"""

from __future__ import annotations

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
    
    def __init__(self, bitstringLength: int = 6, adminAddress: str | None = None,
                 consensusThreshold: int | None = None) -> None:
        """
        Initialize the QRiNG emulator to match smart contract structure
        
        Args:
            bitstringLength (int): Length of bitstrings (matches smart contract array size)
            adminAddress (str): Address of the contract deployer (admin). Set in constructor
                to mirror the Solidity constructor() behavior.
            consensusThreshold (int | None): Minimum matching bits for
                consensus. Defaults to bitstringLength // 2.
        """
        # Smart contract state variables
        self.voters = []  # Array of Voter structs
        self.admin = adminAddress
        self.votingActive = False
        self.initialized = False  # Guard against re-initialization (C5)
        self.counter = []  # 2D array for bitstrings
        self.bitstringLength = bitstringLength
        self.consensusThreshold = consensusThreshold if consensusThreshold is not None else bitstringLength // 2
        
        # Emulation tracking
        self.transactionLog = []
        self.gasConsumption = {}
        self.eventsEmitted = []
        
        # Address simulation (using string addresses like Ethereum)
        self.addresses = []
    
    def _logTransaction(self, functionName: str, caller: str, gasUsed: int,
                        success: bool = True, revertReason: str | None = None) -> None:
        """Log transaction details for emulation tracking"""
        tx = {
            'timestamp': datetime.now().isoformat(),
            'function': functionName,
            'caller': caller,
            'gasUsed': gasUsed,
            'success': success,
            'revertReason': revertReason,
            'blockNumber': len(self.transactionLog) + 1
        }
        self.transactionLog.append(tx)
        
        if functionName not in self.gasConsumption:
            self.gasConsumption[functionName] = []
        self.gasConsumption[functionName].append(gasUsed)
    
    def _emitEvent(self, eventName: str, eventData: dict) -> None:
        """Emulate event emission"""
        event = {
            'event': eventName,
            'data': eventData,
            'blockNumber': len(self.transactionLog) + 1,
            'timestamp': datetime.now().isoformat()
        }
        self.eventsEmitted.append(event)
    
    def addNewString(self, newString: list[list[int]], callerAddress: str) -> bool:
        """
        Emulate addNewString function from smart contract
        function addNewString(uint[][] memory newString) public onlyAdmin
        """
        print(f"Executing addNewString() called by {callerAddress}")
        
        # Gas calculation (approximate)
        rowLen = len(newString[0]) if newString else 0
        gasUsed = 21000 + len(newString) * rowLen * 20  # Base + storage cost
        
        try:
            # Enforce admin-only access control (C4)
            if callerAddress != self.admin:
                raise Exception("Only admin can call this function")

            # Validate all rows are the same length
            if newString:
                expectedLen = len(newString[0])
                for idx, row in enumerate(newString):
                    if len(row) != expectedLen:
                        raise Exception(
                            f"Row {idx} length {len(row)} does not match "
                            f"expected length {expectedLen}"
                        )

            # Store the new bitstrings
            self.counter = [list(bitstring) for bitstring in newString]
            
            self._logTransaction('addNewString', callerAddress, gasUsed, True)
            print(f"✓ Bitstrings stored successfully. Gas used: {gasUsed}")
            return True
            
        except Exception as e:
            self._logTransaction('addNewString', callerAddress, gasUsed, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def setAddresses(self, voterAddresses: list[str], callerAddress: str) -> bool:
        """
        Emulate setAddresses function from smart contract
        function setAddresses(address[] memory voterAddresses) public onlyAdmin
        """
        print(f"Executing setAddresses() called by {callerAddress}")
        
        # Gas calculation based on number of voters to register
        gasUsed = 21000 + len(voterAddresses) * 50000  # Base + voter creation cost
        
        try:
            # Enforce admin-only access control (C5)
            if callerAddress != self.admin:
                raise Exception("Only admin can call this function")

            # Guard against re-initialization (C5)
            if self.initialized:
                raise Exception("Already initialized")
            self.initialized = True

            # Initialize voting system state
            self.votingActive = True
            self.addresses = voterAddresses
            
            # Create voter structs matching Solidity contract structure
            self.voters = []
            for x, address in enumerate(voterAddresses):
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
                    self._emitEvent('VoterRegistered', {'voter': address})
            
            self._logTransaction('setAddresses', callerAddress, gasUsed, True)
            print(f"✓ {len(self.voters)} voters registered. Admin set to {self.admin}")
            print(f"  Gas used: {gasUsed}")
            return True
            
        except Exception as e:
            self._logTransaction('setAddresses', callerAddress, gasUsed, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def endVoting(self, callerAddress: str) -> bool:
        """
        Emulate endVoting function from smart contract
        function endVoting() external onlyAdmin
        """
        print(f"Executing endVoting() called by {callerAddress}")
        
        gasUsed = 21000  # Base gas cost
        
        try:
            # Enforce admin-only access for ending voting
            if callerAddress != self.admin:
                raise Exception("Only admin can call this function")
            
            # Ensure voting is currently active before ending
            if not self.votingActive:
                raise Exception("Voting is not active")
            
            # Deactivate voting system
            self.votingActive = False
            
            # Notify external systems that voting has concluded
            self._emitEvent('VotingEnded', {})
            
            self._logTransaction('endVoting', callerAddress, gasUsed, True)
            print(f"✓ Voting ended. Gas used: {gasUsed}")
            return True
            
        except Exception as e:
            self._logTransaction('endVoting', callerAddress, gasUsed, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def check(self, fromNode: int, callerAddress: str) -> bool:
        """
        Emulate check function from smart contract
        function check(uint from) external onlyActive
        """
        print(f"Executing check({fromNode}) called by {callerAddress}")
        
        # Gas calculation (complex function with loops)
        gasUsed = 21000 + len(self.voters) * len(self.voters[0]['bitstring']) * 10
        
        try:
            # Check onlyActive modifier
            if not self.votingActive:
                raise Exception("Voting is not active")
            
            # Check authorization
            if fromNode >= len(self.voters) or self.voters[fromNode]['delegate'] != callerAddress:
                raise Exception("Not authorized")
            
            # Check if already voted
            if self.voters[fromNode]['hasVoted']:
                raise Exception("Voter has already voted")
              # Perform the check logic (exact replication of Solidity code)
            bitstringThreshold = self.consensusThreshold  # Configurable threshold
            
            # Compare this voter's bitstring with all other voters
            for i in range(len(self.voters)):
                if fromNode != i:
                    add = 0  # Counter for matching bits
                    # Count matching bits between bitstrings
                    for j in range(len(self.voters[fromNode]['bitstring'])):
                        if (self.voters[fromNode]['bitstring'][j] == 
                            self.voters[i]['bitstring'][j]):
                            add += 1
                    
                    # If correlation exceeds threshold, increment vote count
                    if add > bitstringThreshold:
                        self.voters[i]['voteCount'] += 1
            
            # Mark as voted to prevent double-voting
            self.voters[fromNode]['hasVoted'] = True
            
            self._logTransaction('check', callerAddress, gasUsed, True)
            print(f"✓ Check completed for node {fromNode}. Gas used: {gasUsed}")
            return True
            
        except Exception as e:
            self._logTransaction('check', callerAddress, gasUsed, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return False
    
    def getWinner(self, callerAddress: str) -> int | None:
        """
        Emulate getWinner function from smart contract
        function getWinner() external view returns (uint8)
        """
        print(f"Executing getWinner() called by {callerAddress}")
        
        gasUsed = 5000  # View function, lower gas cost
        
        try:
            # Check if voting is still active
            if self.votingActive:
                raise Exception("Voting is still active")
            
            sumCount = 0
            length = len(self.voters) // 2
            
            for i in range(len(self.voters)):
                if self.voters[i]['voteCount'] > length:
                    sumCount += 1
            
            self._logTransaction('getWinner', callerAddress, gasUsed, True)
            print(f"✓ getWinner() returned {sumCount}. Gas used: {gasUsed}")
            return sumCount
            
        except Exception as e:
            self._logTransaction('getWinner', callerAddress, gasUsed, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return None
    
    def randomNumber(self, callerAddress: str) -> list[int] | None:
        """
        Emulate randomNumber function from smart contract
        function randomNumber() external view returns (uint[] memory)
        """
        print(f"Executing randomNumber() called by {callerAddress}")
        
        gasUsed = 10000  # View function with computation
        
        try:            # Check if voting is still active
            if self.votingActive:
                raise Exception("Voting is still active")

            if not self.voters:
                raise Exception("No voters registered")
            
            # Determine bitstring length dynamically from first voter (C6)
            bitstringLength = len(self.voters[0]['bitstring'])
            newBitstring = [0] * bitstringLength
            length = len(self.voters) // 2  # Majority threshold
            
            # XOR operation on honest nodes (C6: use ^= for gas efficiency)
            for i in range(len(self.voters)):
                # Only include voters with sufficient vote count (honest nodes)
                if self.voters[i]['voteCount'] > length:
                    # XOR each bit position
                    for x in range(len(self.voters[i]['bitstring'])):
                        newBitstring[x] ^= self.voters[i]['bitstring'][x]
            
            self._logTransaction('randomNumber', callerAddress, gasUsed, True)
            print(f"✓ randomNumber() returned {newBitstring}. Gas used: {gasUsed}")
            return newBitstring
            
        except Exception as e:
            self._logTransaction('randomNumber', callerAddress, gasUsed, False, str(e))
            print(f"✗ Transaction failed: {e}")
            return None
    
    def getVoterInfo(self, voterIndex: int) -> dict | None:
        """Helper function to get voter information"""
        if 0 <= voterIndex < len(self.voters):
            return self.voters[voterIndex].copy()
        return None
    
    def getContractState(self) -> dict:
        """Get complete contract state for visualization"""
        return {
            'admin': self.admin,
            'votingActive': self.votingActive,
            'numVoters': len(self.voters),
            'voters': [voter.copy() for voter in self.voters],
            'transactionCount': len(self.transactionLog),
            'eventsCount': len(self.eventsEmitted)
        }

    # Deprecation aliases (old snake_case names still work)
    add_new_string = addNewString
    set_addresses = setAddresses
    end_voting = endVoting
    get_winner = getWinner
    random_number = randomNumber
    get_voter_info = getVoterInfo
    get_contract_state = getContractState

def createSmartContractExecutionVisualization(emulator: QRiNGEmulator, save_path: str) -> None:
    """
    Create comprehensive visualization of smart contract execution
    """
    print("Creating smart contract execution visualization...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Transaction timeline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Smart Contract Transaction Timeline", fontsize=16, fontweight='bold')
    
    if emulator.transactionLog:
        timestamps = [i for i in range(len(emulator.transactionLog))]
        functions = [tx['function'] for tx in emulator.transactionLog]
        gasUsed = [tx['gasUsed'] for tx in emulator.transactionLog]
        success = [tx['success'] for tx in emulator.transactionLog]
        
        # Color code by success/failure
        colors = ['green' if s else 'red' for s in success]
        
        bars = ax1.bar(timestamps, gasUsed, color=colors, alpha=0.7, edgecolor='black')
        
        # Add function labels
        for i, (bar, func) in enumerate(zip(bars, functions)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(gasUsed) * 0.01,
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
    
    if emulator.gasConsumption:
        functions = list(emulator.gasConsumption.keys())
        avg_gas = [np.mean(emulator.gasConsumption[func]) for func in functions]
        
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
        bitstringLength = len(emulator.voters[0]['bitstring'])
        
        # Create bitstring matrix
        bitstring_matrix = np.array([voter['bitstring'] for voter in emulator.voters])
        
        im6 = ax6.imshow(bitstring_matrix, cmap='RdYlBu', aspect='auto')
        ax6.set_yticks(range(num_voters))
        ax6.set_yticklabels([f"V{i} (VC:{emulator.voters[i]['voteCount']})" 
                            for i in range(num_voters)])
        ax6.set_xticks(range(bitstringLength))
        ax6.set_xticklabels([f'B{i}' for i in range(bitstringLength)])
        
        # Add bit values
        for i in range(num_voters):
            for j in range(bitstringLength):
                color = 'white' if bitstring_matrix[i, j] == 0 else 'black'
                ax6.text(j, i, f'{bitstring_matrix[i, j]}', 
                        ha='center', va='center', color=color, fontweight='bold')
        
        ax6.set_xlabel('Bit Position')
        ax6.set_ylabel('Voter (Vote Count)')
      # 7. Events and logs
    ax7 = fig.add_subplot(gs[3, 2])
    ax7.set_title("Contract Events", fontsize=14, fontweight='bold')
    ax7.axis('off')
    
    if emulator.eventsEmitted:
        event_text = "Emitted Events:\n\n"
        for i, event in enumerate(emulator.eventsEmitted[-5:]):  # Show last 5 events
            event_text += f"{i+1}. {event['event']}\n"
            if 'voter' in event['data']:
                event_text += f"   Voter: {event['data']['voter'][:10]}...\n"
            event_text += f"   Block: {event['blockNumber']}\n\n"
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

def runEmulationDemo() -> QRiNGEmulator:
    """
    Run a complete demonstration of the QRiNG smart contract emulation
    """
    print("Starting QRiNG Smart Contract Emulation Demo")
    print("=" * 60)
    
    adminAddress = "0x1234567890123456789012345678901234567890"

    # Initialize emulator with admin set in constructor (mirrors Solidity constructor)
    emulator = QRiNGEmulator(bitstringLength=6, adminAddress=adminAddress)
    
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
    
    print("Phase 1: Contract Deployment and Setup")
    print("-" * 40)
    
    # 1. Add bitstrings
    emulator.addNewString(test_bitstrings, adminAddress)
    
    # 2. Set addresses
    emulator.setAddresses(test_addresses, adminAddress)
    
    print("\nPhase 2: Voting Process")
    print("-" * 40)
    
    # 3. Each voter performs check
    for i, address in enumerate(test_addresses):
        emulator.check(i, address)
    
    print("\nPhase 3: Results Generation")
    print("-" * 40)
    
    # 4. End voting
    emulator.endVoting(adminAddress)
      # 5. Get results
    winner_count = emulator.getWinner(adminAddress)
    final_random = emulator.randomNumber(adminAddress)
    
    print(f"\nEMULATION RESULTS:")
    print(f"Honest nodes count: {winner_count}")
    print(f"Final random number: {final_random}")
    
    # Create visualizations
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")
    execution_viz_path = os.path.join(output_dir, "qring_emulator_execution.png")
    
    createSmartContractExecutionVisualization(emulator, execution_viz_path)
    
    print(f"\nVisualization files created:")
    print(f"- Execution visualization: {execution_viz_path}")
    
    return emulator

# Deprecation aliases for standalone functions
create_smart_contract_execution_visualization = createSmartContractExecutionVisualization
run_emulation_demo = runEmulationDemo

if __name__ == "__main__":
    try:
        # Execute the complete smart contract emulation demonstration
        emulator = runEmulationDemo()
        print("\nQRiNG smart contract emulation completed successfully!")
        
        # Display final contract state for verification and analysis
        state = emulator.getContractState()
        print(f"\nFinal Contract State:")
        print(f"- Admin: {state['admin'][:10]}...")      # Show first 10 chars of admin address
        print(f"- Voting Active: {state['votingActive']}")
        print(f"- Number of Voters: {state['numVoters']}")
        print(f"- Total Transactions: {state['transactionCount']}")
        print(f"- Events Emitted: {state['eventsCount']}")
        
    except Exception as e:
        # Handle any errors during emulation execution
        print(f"Error during emulation: {e}")
        import traceback
        traceback.print_exc()
