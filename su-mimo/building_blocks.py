# This module contains the implemtation of the building blocks for the SU-MIMO communication system.


# Import necessary libraries.
import numpy as np
import math



def mapper(bit_array, M, type):
    """
    Convert a bit sequence into data symbols according to the specified modulation constellation.

    Args:
        bit_array (numpy.ndarray): Input bit sequence (2D array of 0s and 1s).
        M (int): Size of the constellation (e.g., 2, 4, 16, 64).
        type (str): Type of constellation ('PAM', 'PSK', or 'QAM').

    Returns:
        symbols (numpy.ndarray): 2D array of complex data symbols.
    """

    
    #1 Divide the input bit sequences into blocks of mc bits, where M = 2^mc.
    
    if bit_array.ndim == 1: bit_array = bit_array[np.newaxis, :]
    Nt, Nbits = bit_array.shape
    mc = int(np.log2(M))
    if (Nbits % mc != 0) : raise ValueError('The length of the bit sequences is invalid. They must be a multiple of log2(M).')    
    
    bit_array = bit_array.flatten()
    bit_array = bit_array.reshape((bit_array.size // mc, mc))


    #2 Convert the blocks of mc bits from gray code to the corresponding decimal value.
    
    graycodes = bit_array
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
        raise ValueError('The constellation type is invalid. It must be "PAM", "PSK", or "QAM".')
    

    #4 Return the data symbol sequences.

    symbols = symbols.reshape((Nt, Nbits // mc))
    return symbols

