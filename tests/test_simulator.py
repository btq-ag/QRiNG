"""Tests for qring.simulator -- QRiNG quantum random number generation."""

import numpy as np
import pytest

from qring.simulator import _HAS_QISKIT, QRiNGSimulator

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestSimulatorInit:
    def testDefaultInit(self):
        sim = QRiNGSimulator(useQuantumBackend=False)
        assert sim.numNodes == 6
        assert sim.bitstringLength == 8
        assert len(sim.bitstrings) == 6

    def testCustomNodeCount(self):
        for n in (2, 5, 10):
            sim = QRiNGSimulator(numNodes=n, useQuantumBackend=False)
            assert sim.numNodes == n
            assert len(sim.bitstrings) == n

    def testCustomBitstringLength(self):
        for length in (4, 16, 32):
            sim = QRiNGSimulator(bitstringLength=length, useQuantumBackend=False)
            assert sim.bitstringLength == length
            for node in sim.nodes:
                assert len(sim.bitstrings[node]) == length

    def testSeedReproducibility(self):
        a = QRiNGSimulator(seed=42, useQuantumBackend=False)
        b = QRiNGSimulator(seed=42, useQuantumBackend=False)
        for node in a.nodes:
            np.testing.assert_array_equal(a.bitstrings[node], b.bitstrings[node])

    def testDifferentSeedsDiffer(self):
        a = QRiNGSimulator(seed=1, useQuantumBackend=False)
        b = QRiNGSimulator(seed=2, useQuantumBackend=False)
        differs = any(not np.array_equal(a.bitstrings[n], b.bitstrings[n]) for n in a.nodes)
        assert differs

    def testNoGlobalRngMutation(self):
        """C3: Constructing a simulator must not touch the global np.random state."""
        np.random.seed(99)
        before = np.random.random()
        np.random.seed(99)
        _ = QRiNGSimulator(seed=0, useQuantumBackend=False)
        after = np.random.random()
        assert before == after


# ---------------------------------------------------------------------------
# Classical bitstring generation
# ---------------------------------------------------------------------------


class TestClassicalBitstrings:
    def testOutputShape(self):
        sim = QRiNGSimulator(numNodes=4, bitstringLength=12, seed=0, useQuantumBackend=False)
        for node in sim.nodes:
            assert sim.bitstrings[node].shape == (12,)

    def testBinaryValues(self):
        sim = QRiNGSimulator(numNodes=6, bitstringLength=64, seed=7, useQuantumBackend=False)
        for node in sim.nodes:
            assert set(sim.bitstrings[node].tolist()).issubset({0, 1})

    def testNotAllIdentical(self):
        sim = QRiNGSimulator(numNodes=6, bitstringLength=64, seed=3, useQuantumBackend=False)
        unique = {tuple(sim.bitstrings[n].tolist()) for n in sim.nodes}
        assert len(unique) > 1


# ---------------------------------------------------------------------------
# Qiskit Aer backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit-aer not installed")
class TestQiskitBackend:
    def testQiskitBitstringsShape(self):
        sim = QRiNGSimulator(numNodes=3, bitstringLength=8, seed=0, useQuantumBackend=True)
        for node in sim.nodes:
            assert sim.bitstrings[node].shape == (8,)

    def testQiskitBinaryValues(self):
        sim = QRiNGSimulator(numNodes=3, bitstringLength=16, seed=0, useQuantumBackend=True)
        for node in sim.nodes:
            assert set(sim.bitstrings[node].tolist()).issubset({0, 1})


def testExplicitQuantumBackendMissingRaises(monkeypatch):
    """Force _HAS_QISKIT=False and request quantum backend."""
    import qring.simulator as mod

    monkeypatch.setattr(mod, "_HAS_QISKIT", False)
    with pytest.raises(RuntimeError, match="qiskit-aer is required"):
        QRiNGSimulator(useQuantumBackend=True)


# ---------------------------------------------------------------------------
# Bitstring similarity
# ---------------------------------------------------------------------------


class TestBitstingSimilarity:
    def testIdenticalBitstrings(self):
        sim = QRiNGSimulator(numNodes=2, bitstringLength=8, seed=0, useQuantumBackend=False)
        # Overwrite so both nodes share the same bitstring
        sim.bitstrings[1] = sim.bitstrings[0].copy()
        assert sim.calculateBitstringSimilarity(0, 1) == 8

    def testComplementBitstrings(self):
        sim = QRiNGSimulator(numNodes=2, bitstringLength=8, seed=0, useQuantumBackend=False)
        sim.bitstrings[0] = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        sim.bitstrings[1] = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        assert sim.calculateBitstringSimilarity(0, 1) == 0

    def testMissingNode(self):
        sim = QRiNGSimulator(numNodes=2, bitstringLength=8, seed=0, useQuantumBackend=False)
        assert sim.calculateBitstringSimilarity(0, 99) == 0


# ---------------------------------------------------------------------------
# Consensus protocol
# ---------------------------------------------------------------------------


class TestConsensus:
    def testPerformConsensusCheckMarksVoted(self):
        sim = QRiNGSimulator(numNodes=4, bitstringLength=8, seed=0, useQuantumBackend=False)
        assert sim.performConsensusCheck(0) is True
        assert sim.hasVoted[0]

    def testDoubleVotePrevented(self):
        sim = QRiNGSimulator(numNodes=4, bitstringLength=8, seed=0, useQuantumBackend=False)
        sim.performConsensusCheck(0)
        assert sim.performConsensusCheck(0) is False

    def testAllNodesVote(self):
        sim = QRiNGSimulator(numNodes=4, bitstringLength=8, seed=0, useQuantumBackend=False)
        sim.runConsensusProtocol()
        assert all(sim.hasVoted)

    def testHonestNodesIdentified(self):
        """Force correlated bitstrings so most nodes pass consensus."""
        sim = QRiNGSimulator(numNodes=6, bitstringLength=8, seed=0, useQuantumBackend=False)
        # Force nodes to share similar bitstrings so consensus succeeds
        base = sim.bitstrings[0].copy()
        for node in range(1, sim.numNodes):
            sim.bitstrings[node] = base.copy()
        honest = sim.runConsensusProtocol()
        assert isinstance(honest, list)
        assert len(honest) >= 1

    def testKnownDishonestScenario(self):
        """Inject a clearly adversarial node and verify it is excluded."""
        sim = QRiNGSimulator(numNodes=4, bitstringLength=16, seed=0, useQuantumBackend=False)
        # Make node 3 wildly different
        sim.bitstrings[3] = 1 - sim.bitstrings[0]
        sim.runConsensusProtocol()
        # Node 3 should have low vote count
        assert sim.voteCounts[3] <= sim.numNodes // 2


# ---------------------------------------------------------------------------
# Final random number
# ---------------------------------------------------------------------------


class TestFinalRandomNumber:
    def testRunFullSimulation(self):
        sim = QRiNGSimulator(numNodes=6, bitstringLength=8, seed=0, useQuantumBackend=False)
        # Force correlated bitstrings to ensure honest nodes exist
        base = sim.bitstrings[0].copy()
        for node in range(1, sim.numNodes):
            sim.bitstrings[node] = base.copy()
        result = sim.runFullSimulation()
        assert result["final_random_number"] is not None
        assert len(result["final_random_number"]) == 8

    def testOutputIsBinary(self):
        sim = QRiNGSimulator(numNodes=4, bitstringLength=16, seed=0, useQuantumBackend=False)
        sim.runConsensusProtocol()
        bits = sim.generateFinalRandomNumber()
        if bits is not None:
            assert set(bits.tolist()).issubset({0, 1})

    def testNoHonestNodesReturnsNone(self):
        sim = QRiNGSimulator(numNodes=2, bitstringLength=8, seed=0, useQuantumBackend=False)
        # Make bitstrings maximally different so no one passes consensus
        sim.bitstrings[0] = np.zeros(8, dtype=int)
        sim.bitstrings[1] = np.ones(8, dtype=int)
        sim.runConsensusProtocol()
        result = sim.generateFinalRandomNumber()
        assert result is None

    def testXorAggregation(self):
        """Verify the XOR logic directly."""
        sim = QRiNGSimulator(numNodes=3, bitstringLength=4, seed=0, useQuantumBackend=False)
        sim.bitstrings[0] = np.array([1, 0, 1, 0])
        sim.bitstrings[1] = np.array([0, 1, 1, 0])
        sim.bitstrings[2] = np.array([1, 1, 0, 0])
        sim.honestNodes = [0, 1, 2]
        bits = sim.generateFinalRandomNumber()
        expected = np.array([1, 0, 1, 0]) ^ np.array([0, 1, 1, 0]) ^ np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(bits, expected)


# ---------------------------------------------------------------------------
# Configurable consensus threshold
# ---------------------------------------------------------------------------


class TestConsensusThreshold:
    def testDefaultThreshold(self):
        sim = QRiNGSimulator(numNodes=4, bitstringLength=8, seed=0, useQuantumBackend=False)
        assert sim.consensusThreshold == 4  # bitstringLength // 2

    def testCustomThreshold(self):
        sim = QRiNGSimulator(
            numNodes=4, bitstringLength=8, seed=0, useQuantumBackend=False, consensusThreshold=6
        )
        assert sim.consensusThreshold == 6

    def testHighThresholdReducesHonestNodes(self):
        """A higher threshold should make it harder to pass consensus."""
        sim_low = QRiNGSimulator(
            numNodes=6, bitstringLength=8, seed=42, useQuantumBackend=False, consensusThreshold=2
        )
        sim_high = QRiNGSimulator(
            numNodes=6, bitstringLength=8, seed=42, useQuantumBackend=False, consensusThreshold=7
        )
        # Copy same bitstrings
        for n in sim_high.nodes:
            sim_high.bitstrings[n] = sim_low.bitstrings[n].copy()

        honest_low = sim_low.runConsensusProtocol()
        honest_high = sim_high.runConsensusProtocol()
        assert len(honest_low) >= len(honest_high)
