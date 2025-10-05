# This module contains the test functions of the building blocks for the SU-MIMO communication system.

# Import necessary libraries.
import building_blocks
import numpy as np


def test_mapper_demapper(mapper, demapper):
    """
    Test mapper and demapper functions for multiple different constellations and MIMO streams.

    Args:
        mapper (function): Mapper function from building_blocks module.
        demapper (function): Demapper function from building_blocks module.
    """

    print("=== Testing Mapper & Demapper ===\n")

    # Define modulation configurations.
    configs = [ {"M": 4, "type": "PAM"}, {"M": 8, "type": "PSK"}, {"M": 16, "type": "QAM"}, {"M": 64, "type": "QAM"}]
    Nt = 2
    Nbits = 480


    for cfg in configs:
        
        # Generate random bits for all antennas, map them to data symbols, and demap them back to bits.
        bits_tx = np.random.randint(0, 2, (Nt, Nbits))
        symbols = mapper(bits_tx, cfg["M"], cfg["type"])
        bits_rx = demapper(symbols, cfg["M"], cfg["type"])

        # Compute the bit error rate and the average symbol power.
        ber = np.sum(bits_tx != bits_rx) / bits_tx.size
        avg_power = np.mean(np.abs(symbols)**2)

        # Display results
        print(f"Modulation: {cfg["type"]} (M={cfg["M"]})")
        print(f"  Bits per symbol (mc): {int(np.log2(cfg["M"]))}")
        print(f"  Output symbols shape: {symbols.shape}")
        print(f"  Average symbol power: {avg_power:.3f}")
        print(f"  Bit Error Rate (BER): {ber:.3e}")
        print(f"  Test {'✅ PASSED' if ber == 0 else '❌ FAILED'}\n")


    print("=== All tests completed ===")



# Run the test function.
test_mapper_demapper(building_blocks.mapper, building_blocks.demapper)