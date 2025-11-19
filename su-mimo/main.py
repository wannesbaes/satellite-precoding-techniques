# In this file, we perform various tests on the SU-MIMO SVD digital communication system. 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.su_mimo import SuMimoSVD
from src.transmitter import Transmitter
from src.channel import Channel
from src.receiver import Receiver


test = 'test 0'
    
# TEST 0: Example simulation of SU-MIMO SVD system.
if test == 'example':

    su_mimo_svd = SuMimoSVD(Nt=3, Nr=2, c_type='PSK', H=np.array([[3, 2, 2], [2, 3, -2]]))
    ber = su_mimo_svd.print_simulation_example(bits=np.random.randint(0, 2, size=16000), K=3)

# TEST 1: Performance analysis through bit error rate curves.
if test == 'BER':

    # 1. Analyse the performance for different SU-MIMO SVD DigCom Systems.
    # 2. Analyse the performance of the different eigenchannels.
    # 3. Analyse the performance at different data rates.
    pass

# TEST 2: Performance analysis through scatter diagrams.
if test == 'scatter':
    
    # 1. Analyse the scatter diagram at different SNR values (for the same SU-MIMO SVD DigCom System).
    # 2. Analyse the scatter diagram at different data rates (for the same SU-MIMO SVD DigCom System).
    pass

