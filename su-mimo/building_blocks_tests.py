# This module contains the test functions of the building blocks for the SU-MIMO communication system.

# Import necessary libraries.

import numpy as np
import su_mimo


def test_mapper_demapper():
    """ Test mapper and demapper functions for multiple different constellations and MIMO streams."""

    # Define modulation constellation configurations.
    configs = [ {"M": 4, "type": "PAM"}, {"M": 8, "type": "PSK"}, {"M": 16, "type": "QAM"}, {"M": 64, "type": "QAM"}]


    for cfg in configs:

        # Initialize the communication system with the given configuration and generate random bits.
        su_mimo_svd_digcomsys = su_mimo.SuMimoSVD(Nt=2, Nr=2, constellation_size=cfg["M"], constellation_type=cfg["type"], SNR=0)
        su_mimo_svd_digcomsys._bits = np.random.randint(0, 2, (2, 480))
        
        # Map the random bits to symbols and then demap them back to bits. Assume an ideal channel (no noise, no fading) for testing.
        su_mimo_svd_digcomsys.mapper()
        su_mimo_svd_digcomsys._symbols_hat = su_mimo_svd_digcomsys._symbols
        su_mimo_svd_digcomsys.demapper()

        # Compute the bit error rate and the average symbol power.
        ber = su_mimo_svd_digcomsys.ber()
        avg_power = np.mean(np.abs(su_mimo_svd_digcomsys._symbols)**2)

        # Display results
        print(f"Modulation: {cfg["type"]} (M={cfg["M"]})")
        print(f"  Bits per symbol (mc): {int(np.log2(cfg["M"]))}")
        print(f"  Output symbols shape: {su_mimo_svd_digcomsys._symbols.shape}")
        print(f"  Average symbol power: {avg_power:.3f}")
        print(f"  Bit Error Rate (BER): {ber:.3e}")
        print(f"  Test {'✅ PASSED' if ber == 0 else '❌ FAILED'}\n")


# Run the test function.
test_mapper_demapper()