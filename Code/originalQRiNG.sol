// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
 
contract SmartContract {
    
    // Voter structure containing delegate address, voting status, and quantum bitstring
    struct Voter {
        address delegate;        // Address authorized to vote on behalf of this voter
        bool hasVoted;          // Flag to track if voter has already participated
        uint number;            // Unique identifier for this voter
        uint voteCount;         // Number of votes received from other voters
        uint[] bitstring;       // Quantum-generated bitstring for this voter
    }

    Voter[] public voters;          // Dynamic array of all voters in the system
    address public admin;           // Administrator address with special permissions
    bool public votingActive;       // Flag to control voting phase activation
    uint[][] public counter;        // 2D array storing all quantum bitstrings

    // Events for external monitoring
    event VoterRegistered(address voter);   // Emitted when new voter is added
    event VotingEnded();                   // Emitted when voting phase ends

    // Modifier to restrict function access to admin only
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can call this function");
        _;
    }

    // Modifier to ensure voting is currently active
    modifier onlyActive() {
        require(votingActive, "Voting is not active");
        _;
    }

    // Store new quantum bitstrings in the counter array
    function addNewString(uint[][] memory newString) public { 
        counter = newString;
    }

    // Initialize the voting system with voter addresses and their quantum bitstrings
    function setAddresses(address[] memory voterAddresses) public {
        admin = msg.sender;     // Set caller as admin
        votingActive = true;    // Activate voting phase
        
        // Create voter entries with assigned bitstrings
        for (uint x = 0; x < voterAddresses.length; x++) {
            voters.push(Voter({
                delegate: voterAddresses[x],
                hasVoted: false,
                number: x,
                voteCount: 0,
                bitstring: counter[x]  // Assign pre-stored quantum bitstring
            }));
            emit VoterRegistered(voterAddresses[x]);
        }
    }

    // Start the voting process (admin only)
    function startVoting() external onlyAdmin {
        require(!votingActive, "Voting is already active");
        votingActive = true;
    }

    // End the voting process and allow result computation (admin only)
    function endVoting() external onlyAdmin {
        require(votingActive, "Voting is not active");
        votingActive = false;
        emit VotingEnded();
    }

    // Validate quantum measurements by comparing bitstrings between voters
    function check(uint from) external onlyActive {
        require(voters[from].delegate == msg.sender, "Not authorized");
        require(!voters[from].hasVoted, "Voter has already voted");
        
        // Calculate threshold for quantum correlation (half of bitstring length)
        uint bitstringLength = voters[from].bitstring.length / 2;
        
        // Compare this voter's bitstring with all other voters
        for (uint i = 0; i < voters.length; i++) {
            if (from != i) {
                uint add = 0;  // Counter for matching bits
                
                // Count matching bits between bitstrings
                for (uint j = 0; j < voters[from].bitstring.length; j++) {
                    if (voters[from].bitstring[j] == voters[i].bitstring[j]) {
                        add++;
                    }
                }
                
                // If correlation exceeds threshold, increment vote count
                if (add > bitstringLength) {
                    voters[i].voteCount++;
                }
            }
        }
        voters[from].hasVoted = true;  // Mark as voted to prevent double-voting
    }

    // Count how many voters achieved consensus (vote count above threshold)
    function getWinner() external view returns (uint8) {
        require(!votingActive, "Voting is still active");

        uint8 sum = 0;  // Counter for voters with sufficient votes
        uint256 len = voters.length / 2;  // Majority threshold
        
        // Count voters that exceeded the vote threshold
        for (uint i = 0; i < voters.length; i++) {
            if (voters[i].voteCount > len) {
                sum++;
            }
        }
        
        return sum;  // Return number of consensus-achieving voters
    }

    // Generate final random number by XOR-ing bitstrings of consensus voters
    function randomNumber() external view returns (uint[6] memory newBitstring) {
        require(!votingActive, "Voting is still active");
        uint256 len = voters.length / 2;  // Majority threshold

        // XOR together bitstrings from voters with sufficient vote counts
        for (uint i = 0; i < voters.length; i++) {
            if (voters[i].voteCount > len) {
                // XOR each bit position with the corresponding bit from this voter
                for (uint x = 0; x < voters[i].bitstring.length; x++) {
                    newBitstring[x] += voters[i].bitstring[x];
                    newBitstring[x] %= 2;  // Ensure binary result (0 or 1)
                }
            }
        }
        
        return newBitstring;  // Return final quantum random bitstring
    }
}
