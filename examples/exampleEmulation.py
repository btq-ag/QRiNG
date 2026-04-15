"""
QRiNG Example: Smart Contract Emulation

Demonstrates the emulator replicating the Solidity contract logic.
"""

from qring import QRiNGEmulator

admin = "0xADMIN"
emu = QRiNGEmulator(bitstringLength=6, adminAddress=admin)

bitstrings = [
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0],
]
addresses = ["0xA", "0xB", "0xC"]

emu.addNewString(bitstrings, admin)
emu.setAddresses(addresses, admin)

for i, addr in enumerate(addresses):
    emu.check(i, addr)

emu.endVoting(admin)

winner = emu.getWinner(admin)
randomBits = emu.randomNumber(admin)

print(f"Honest node count: {winner}")
print(f"Random number: {randomBits}")
