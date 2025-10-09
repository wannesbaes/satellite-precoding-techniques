import numpy as np
from su_mimo import SuMimoSVD, Transmitter, Channel, Receiver

# Set random seed for reproducibility
# np.random.seed(42)

# Test Configuration
Nt = 4  # Number of transmit antennas
Nr = 4  # Number of receive antennas
M = 16  # Constellation size (4-QAM, 16-QAM, etc.)
constellation_type = 'QAM'  # 'PAM', 'PSK', or 'QAM'
SNR = 30  # Signal-to-noise ratio in dB
Nbits = 64  # Number of bits per antenna

print("="*60)
print("SU-MIMO SVD Communication System Test")
print("="*60)

# Create the communication system
system = SuMimoSVD(Nt, Nr, M, constellation_type, SNR, Nbits)
print(system)
print()

# Display the transmitted bits
print("Transmitted bits (first antenna, first 20 bits):")
print(system._bits[0, :20])
print()

# Run the simulation
print("Running simulation...")
system.simulate()
print("✓ Simulation complete!")
print()

# Display results
print("Received bits (first antenna, first 20 bits):")
print(system._bits_hat[0, :20])
print()

# Calculate Bit Error Rate (BER)
bit_errors = np.sum(system._bits != system._bits_hat)
total_bits = system._bits.size
BER = bit_errors / total_bits

print(f"Performance Metrics:")
print(f"  Total bits transmitted: {total_bits}")
print(f"  Bit errors: {bit_errors}")
print(f"  Bit Error Rate (BER): {BER:.6f}")
print()

# Check if communication was perfect
if bit_errors == 0:
    print("✓ Perfect transmission! All bits received correctly.")
else:
    print(f"✗ {bit_errors} bit errors detected.")

print()

# Test with different SNR values
# print("="*60)
# print("BER vs SNR Test")
# print("="*60)
# SNR_values = [0, 5, 10, 15, 20, 25, 30]
# print(f"{'SNR (dB)':<10} {'BER':<15} {'Bit Errors':<12}")
# print("-"*40)

# for snr in SNR_values:
#     # Create new system with same bits for fair comparison
#     test_system = SuMimoSVD(Nt, Nr, M, constellation_type, snr, Nbits, bits=system._bits)
#     test_system.simulate()
    
#     errors = np.sum(test_system._bits != test_system._bits_hat)
#     ber = errors / total_bits
    
#     print(f"{snr:<10} {ber:<15.6e} {errors:<12}")

# print("="*60)