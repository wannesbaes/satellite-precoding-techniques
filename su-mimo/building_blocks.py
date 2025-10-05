# This module contains the implemtation of the building blocks for the SU-MIMO communication system.


# Import necessary libraries.
import numpy as np
import math



def mapper(bits, M, type):
    """
    Convert bit sequences into data symbol sequences according to the specified modulation constellation.

    Args:
        bits (numpy.ndarray): Input bit sequences (2D array of 0s and 1s).
        M (int): Size of the constellation (e.g., 2, 4, 16, 64).
        type (str): Type of constellation ('PAM', 'PSK', or 'QAM').

    Returns:
        symbols (numpy.ndarray): Output sequences of complex data symbols.
    """

    
    #1 Divide the input bit sequences into blocks of mc bits, where M = 2^mc.
    
    if bits.ndim == 1: bits = bits[np.newaxis, :]
    Nt, Nbits = bits.shape
    mc = int(np.log2(M))
    if (Nbits % mc != 0) : raise ValueError('The length of the bit sequences is invalid. They must be a multiple of log2(M).')    
    
    bits = bits.flatten()
    bits = bits.reshape((bits.size // mc, mc))


    #2 Convert the blocks of mc bits from gray code to the corresponding decimal value.
    
    graycodes = bits
    binarycodes = np.zeros_like(graycodes)
    binarycodes[:, 0] = graycodes[:, 0]

    for i in range(1, graycodes.shape[1]):
        binarycodes[:, i] = binarycodes[:, i-1] ^ graycodes[:, i]
    
    decimals = np.dot(binarycodes, (2**np.arange(mc))[::-1])  
   

    #3 Convert the decimal values to the corresponding data symbols, according to the specified constellation type.

    if type == 'PAM' :
        dmin = math.sqrt(12/(M**2-1))
        symbols = (decimals - (M-1)/2) * dmin
    
    elif type == 'PSK' :
        symbols = np.exp(2 * np.pi * decimals * 1j / M)

    elif type == 'QAM' :
        if (mc % 2 != 0) : raise ValueError('The constellation size M is invalid. For QAM Modulation, M must be a power of 4 (e.g., 4, 16, 64).')
        dmin = math.sqrt(6/(M-1))
        symbols_real_part = ((decimals//math.sqrt(M)) - (math.sqrt(M)-1)/2) * dmin
        symbols_imaginary_part = ( np.where( ((decimals//int(math.sqrt(M))) % 2 == 0), (int(math.sqrt(M))-1) - (decimals % int(math.sqrt(M))), (decimals % int(math.sqrt(M))) ) - (int(math.sqrt(M))-1)/2) * dmin
        symbols = symbols_real_part + (symbols_imaginary_part * 1j)

    else :
        raise ValueError('The constellation type is invalid. Choose between "PAM", "PSK", or "QAM".')
    

    #4 Return the output data symbol sequences.

    symbols = symbols.reshape((Nt, Nbits // mc))
    return symbols

def demapper(symbols, M, type):
    """
    Convert data symbol sequences into bit sequences according to the specified modulation constellation.

    Args:
        symbols (numpy.ndarray): Input data symbol sequence (2D array of complex symbols).
        M (int): Size of the constellation (e.g., 2, 4, 16, 64).
        type (str): Type of constellation ('PAM', 'PSK', or 'QAM').
    
    Returns:
        bits (numpy.ndarray): Output sequences of bits.
    """

    #1 Setup.

    mc = int(np.log2(M))
    if symbols.ndim == 1: symbols = symbols[np.newaxis, :]
    Nt, Nsymbols = symbols.shape
    symbols = symbols.flatten()


    #2 Convert the data symbols to the corresponding decimal values, according to the specified constellation type.
    
    if type == 'PAM':
        dmin = math.sqrt(12/(M**2-1))
        decimals = np.round(symbols/dmin + (M-1)/2).astype(int)
    
    elif type == 'PSK':
        phases = np.angle(symbols)
        phases[phases < 0] += 2*np.pi
        decimals = np.round((phases * M) / (2*np.pi)).astype(int)
    
    elif type == 'QAM':
        if (mc % 2 != 0): raise ValueError('The constellation size M is invalid. For QAM Modulation, M must be a power of 4 (e.g., 4, 16, 64).')
        dmin = math.sqrt(6/(M-1))
        real_parts = np.round( np.real(symbols)/dmin + (int(math.sqrt(M))-1)/2 ).astype(int)
        imaginary_parts = np.round( np.imag(symbols)/dmin + (int(math.sqrt(M))-1)/2 ).astype(int)
        decimals = (real_parts * int(math.sqrt(M))) + np.where((real_parts % 2 == 0), (int(math.sqrt(M))-1) - imaginary_parts, imaginary_parts)

    else:
        raise ValueError('The constellation type is invalid. Choose between "PAM", "PSK", or "QAM".')


    #3 Convert the decimal values to the corresponding blocks of mc bits in gray code.

    binarycodes = ((decimals[:, None].astype(int) & (1 << np.arange(mc))[::-1].astype(int)) > 0).astype(int)

    graycodes = np.zeros_like(binarycodes)
    graycodes[:, 0] = binarycodes[:, 0]
    for i in range(1, mc):
        graycodes[:, i] = binarycodes[:, i] ^ binarycodes[:, i - 1]


    #4 Return the output bit sequences.

    bits = graycodes.flatten()
    bits = bits.reshape((Nt, Nsymbols * mc))
    return bits
