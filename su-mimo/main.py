# In this file, we perform various tests on the SU-MIMO SVD digital communication system. 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.su_mimo import SuMimoSVD
from src.transmitter import Transmitter
from src.channel import Channel
from src.receiver import Receiver


test = ''

# TEST 0: Example simulation of SU-MIMO SVD system.
if test == 'example':

    RAS = {'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': 1.0, 'constellation sizes': None, 'control channel': True}
    SNR = 10
    H = np.array([[3, 2, 2], [2, 3, -2]])

    su_mimo_svd = SuMimoSVD(Nt=3, Nr=2, c_type='QAM', Pt=1.0, B=0.5, RAS=RAS, SNR=SNR, H=H)
    ber = su_mimo_svd.print_simulation_example(bitstream=np.random.randint(0, 2, size=2000), K=3)

# TEST 1: Performance analysis through bit error rate curves.
if test == 'BER':

    # 1. Analyse the performance for different SU-MIMO SVD DigCom Systems.
    # 2. Analyse the performance of the different eigenchannels.
    # 3. Analyse the performance at different data rates.
    pass

# TEST 2: Performance analysis through scatter diagrams.
if test == 'scatter':

    # 1. Analyse the scatter diagram at different SNR values, for the same SU-MIMO SVD DigCom System with the same information bit rate.
    # 2. Analyse the scatter diagram at different data rates, for the same SU-MIMO SVD DigCom System and the same SNR value.
    # 3. Analyse the scatter diagram for different SU-MIMO SVD DigCom Systems, for the same SNR value and information bit rate.
    pass
    
# TEST 3: Performance analysis through analytical SNR curves.
if test == 'analytical':

    # 1. Analyse the Bit Error Rate (BER) vs SNR performance at different data rates, for the same SU-MIMO SVD DigCom System.
    pass