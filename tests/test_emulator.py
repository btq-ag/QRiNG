"""Tests for qring.emulator -- QRiNG smart contract emulation."""

from qring.emulator import QRiNGEmulator

ADMIN = "0xADMIN_000000000000000000000000000000000"
OTHER = "0xOTHER_000000000000000000000000000000000"
ADDRESSES = [f"0x{i:040x}" for i in range(4)]
BITSTRINGS = [
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 0, 1],
]


def _setup_ready_emulator():
    """Return an emulator with bitstrings loaded and addresses set."""
    emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
    emu.addNewString(BITSTRINGS, ADMIN)
    emu.setAddresses(ADDRESSES, ADMIN)
    return emu


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestEmulatorInit:
    def testDefaultInit(self):
        emu = QRiNGEmulator(adminAddress=ADMIN)
        assert emu.admin == ADMIN
        assert emu.votingActive is False
        assert emu.initialized is False
        assert emu.voters == []

    def testAdminSetInConstructor(self):
        """C5: admin is set at construction time, not in setAddresses."""
        emu = QRiNGEmulator(adminAddress=ADMIN)
        assert emu.admin == ADMIN


# ---------------------------------------------------------------------------
# addNewString
# ---------------------------------------------------------------------------


class TestAddNewString:
    def testValidInput(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        assert emu.addNewString(BITSTRINGS, ADMIN) is True
        assert len(emu.counter) == 4

    def testNonAdminRejected(self):
        """C4: Only admin can call addNewString."""
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        assert emu.addNewString(BITSTRINGS, OTHER) is False

    def testOverwriteAllowed(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        emu.addNewString(BITSTRINGS, ADMIN)
        new = [[0, 0, 0, 0, 0, 0]]
        assert emu.addNewString(new, ADMIN) is True
        assert len(emu.counter) == 1


# ---------------------------------------------------------------------------
# setAddresses
# ---------------------------------------------------------------------------


class TestSetAddresses:
    def testValidSetup(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        emu.addNewString(BITSTRINGS, ADMIN)
        assert emu.setAddresses(ADDRESSES, ADMIN) is True
        assert len(emu.voters) == 4
        assert emu.votingActive is True

    def testNonAdminRejected(self):
        """C5: Only admin can call setAddresses."""
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        emu.addNewString(BITSTRINGS, ADMIN)
        assert emu.setAddresses(ADDRESSES, OTHER) is False

    def testDoubleInitRejected(self):
        """C5: setAddresses cannot be called twice."""
        emu = _setup_ready_emulator()
        assert emu.setAddresses(ADDRESSES, ADMIN) is False

    def testCorrectNumberOfAddresses(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        emu.addNewString(BITSTRINGS, ADMIN)
        # Fewer addresses than bitstrings is fine (only creates matching voters)
        short = ADDRESSES[:2]
        assert emu.setAddresses(short, ADMIN) is True
        assert len(emu.voters) == 2


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


class TestCheck:
    def testValidCheck(self):
        emu = _setup_ready_emulator()
        assert emu.check(0, ADDRESSES[0]) is True
        assert emu.voters[0]["hasVoted"] is True

    def testDoubleCheckRejected(self):
        emu = _setup_ready_emulator()
        emu.check(0, ADDRESSES[0])
        assert emu.check(0, ADDRESSES[0]) is False

    def testUnauthorizedCallerRejected(self):
        emu = _setup_ready_emulator()
        assert emu.check(0, OTHER) is False

    def testMatchingBitstringsIncrementVotes(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        identical = [[1, 0, 1, 1, 0, 1]] * 4
        emu.addNewString(identical, ADMIN)
        emu.setAddresses(ADDRESSES, ADMIN)
        emu.check(0, ADDRESSES[0])
        # All other nodes should have gotten a vote
        for i in range(1, 4):
            assert emu.voters[i]["voteCount"] >= 1

    def testNonMatchingBitstringsNoVotes(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        opposite = [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ]
        emu.addNewString(opposite, ADMIN)
        emu.setAddresses(ADDRESSES, ADMIN)
        emu.check(0, ADDRESSES[0])
        # Nodes 1 and 3 are complements of node 0, should not get votes
        assert emu.voters[1]["voteCount"] == 0
        assert emu.voters[3]["voteCount"] == 0


# ---------------------------------------------------------------------------
# endVoting + getWinner
# ---------------------------------------------------------------------------


class TestEndVotingAndGetWinner:
    def testEndVotingFlow(self):
        emu = _setup_ready_emulator()
        for i, addr in enumerate(ADDRESSES):
            emu.check(i, addr)
        assert emu.endVoting(ADMIN) is True
        assert emu.votingActive is False

    def testEndVotingNonAdminRejected(self):
        emu = _setup_ready_emulator()
        assert emu.endVoting(OTHER) is False

    def testGetWinnerCountsHonestNodes(self):
        emu = _setup_ready_emulator()
        for i, addr in enumerate(ADDRESSES):
            emu.check(i, addr)
        emu.endVoting(ADMIN)
        count = emu.getWinner(ADMIN)
        assert isinstance(count, int)
        assert count >= 0

    def testGetWinnerWhileVotingRejected(self):
        emu = _setup_ready_emulator()
        assert emu.getWinner(ADMIN) is None


# ---------------------------------------------------------------------------
# randomNumber
# ---------------------------------------------------------------------------


class TestRandomNumber:
    def testFullFlow(self):
        emu = _setup_ready_emulator()
        for i, addr in enumerate(ADDRESSES):
            emu.check(i, addr)
        emu.endVoting(ADMIN)
        result = emu.randomNumber(ADMIN)
        assert isinstance(result, list)
        assert all(b in (0, 1) for b in result)

    def testXorAggregation(self):
        """Verify XOR logic matches expected output."""
        emu = QRiNGEmulator(bitstringLength=4, adminAddress=ADMIN)
        bitstrings = [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ]
        addrs = ADDRESSES[:2]
        emu.addNewString(bitstrings, ADMIN)
        emu.setAddresses(addrs, ADMIN)
        # Make both nodes "honest" by giving them high vote counts
        for v in emu.voters:
            v["voteCount"] = 10
            v["hasVoted"] = True
        emu.votingActive = False
        result = emu.randomNumber(ADMIN)
        # XOR: [1^0, 0^1, 1^1, 0^0] = [1, 1, 0, 0]
        assert result == [1, 1, 0, 0]

    def testWhileVotingRejected(self):
        emu = _setup_ready_emulator()
        assert emu.randomNumber(ADMIN) is None

    def testDynamicBitstringLength(self):
        """C6: randomNumber uses dynamic length from voter bitstrings."""
        emu = QRiNGEmulator(bitstringLength=10, adminAddress=ADMIN)
        long_bits = [[1] * 10, [0] * 10]
        addrs = ADDRESSES[:2]
        emu.addNewString(long_bits, ADMIN)
        emu.setAddresses(addrs, ADMIN)
        for v in emu.voters:
            v["voteCount"] = 10
            v["hasVoted"] = True
        emu.votingActive = False
        result = emu.randomNumber(ADMIN)
        assert len(result) == 10

    def testNoVotersRejected(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        emu.votingActive = False
        assert emu.randomNumber(ADMIN) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def testZeroVoters(self):
        """C1 audit: 0 voters edge case."""
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        emu.addNewString([], ADMIN)
        # set_addresses with empty list
        assert emu.setAddresses([], ADMIN) is True
        assert len(emu.voters) == 0

    def testSingleVoter(self):
        """C1 audit: 1 voter edge case."""
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        emu.addNewString([[1, 0, 1, 0, 1, 0]], ADMIN)
        emu.setAddresses([ADDRESSES[0]], ADMIN)
        assert len(emu.voters) == 1
        # Single voter checks against no others, so no votes are cast
        emu.check(0, ADDRESSES[0])
        assert emu.voters[0]["voteCount"] == 0

    def testEmptyBitstrings(self):
        """C1 audit: empty bitstrings edge case."""
        emu = QRiNGEmulator(bitstringLength=0, adminAddress=ADMIN)
        emu.addNewString([[], []], ADMIN)
        emu.setAddresses(ADDRESSES[:2], ADMIN)
        assert len(emu.voters) == 2

    def testContractStateSnapshot(self):
        emu = _setup_ready_emulator()
        state = emu.getContractState()
        assert state["admin"] == ADMIN
        assert state["votingActive"] is True
        assert state["numVoters"] == 4


# ---------------------------------------------------------------------------
# Row-length validation
# ---------------------------------------------------------------------------


class TestRowLengthValidation:
    def testJaggedBitstringsRejected(self):
        """addNewString should reject rows of different lengths."""
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        jagged = [[1, 0, 1], [1, 0]]  # different lengths
        assert emu.addNewString(jagged, ADMIN) is False

    def testUniformBitstringsAccepted(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        uniform = [[1, 0, 1], [0, 1, 0]]
        assert emu.addNewString(uniform, ADMIN) is True


# ---------------------------------------------------------------------------
# Configurable consensus threshold
# ---------------------------------------------------------------------------


class TestConsensusThreshold:
    def testDefaultThreshold(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN)
        assert emu.consensusThreshold == 3  # bitstringLength // 2

    def testCustomThreshold(self):
        emu = QRiNGEmulator(bitstringLength=6, adminAddress=ADMIN, consensusThreshold=5)
        assert emu.consensusThreshold == 5
