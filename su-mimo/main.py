# In this file, we perform various tests on the SU-MIMO SVD digital communication system. 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.su_mimo import SuMimoSVD
from src.transmitter import Transmitter
from src.channel import Channel
from src.receiver import Receiver


test = 'analytical'
    
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

    H = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)

    # 1. Analyse the scatter diagram at different SNR values, for the same SU-MIMO SVD DigCom System with the same information bit rate.
    for SNR in [5, 10, 15]:
        su_mimo_svd = SuMimoSVD(Nt=4, Nr=4, c_type='QAM', H=H, SNR=SNR)
        su_mimo_svd.plot_scatter_diagram()
        plt.show()
    
    # 2. Analyse the scatter diagram at different data rates, for the same SU-MIMO SVD DigCom System and the same SNR value.
    for data_rate in [0.5, 0.75, 1.0]:
        su_mimo_svd = SuMimoSVD(Nt=4, Nr=4, c_type='QAM', H=H, SNR=15, data_rate=data_rate)
        su_mimo_svd.plot_scatter_diagram()
        plt.show()
    
    # 3. Analyse the scatter diagram for different SU-MIMO SVD DigCom Systems, for the same SNR value and information bit rate.
    for Nt, Nr in [(2, 2), (4, 2), (4, 4)]:
        su_mimo_svd = SuMimoSVD(Nt=Nt, Nr=Nr, c_type='QAM', SNR=15)
        su_mimo_svd.plot_scatter_diagram()
        plt.show()

# TEST 3: Performance analysis through analytical SNR curves.
if test == 'analytical':

    # 1. Analyse the Bit Error Rate (BER) vs SNR performance at different data rates, for the same SU-MIMO SVD DigCom System.
    SNRs = np.arange(0, 26, 2.5)

    labels, BERs, data_Rs, activation_Rs = [], [], [], []
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:red', 'tab:purple', 'tab:red']
    markers = ['o', 'o', '^', '^', 'x', 'x']

    for data_rate in [0.5, 1.0]:
        
        su_mimo_svd = SuMimoSVD(Nt=4, Nr=4, c_type='PSK', data_rate=data_rate)
        
        BERs_sim, data_Rs_sim, activation_Rs_sim = su_mimo_svd.BERs_simulation(SNRs=SNRs, num_errors=200, num_channels=100)
        BERs_analytical, data_Rs_analytical, activation_Rs_analytical = su_mimo_svd.BERs_analytical(SNRs=SNRs, num_channels=100, settings={'mode': 'approximation', 'eigenchannels': False})
        
        labels.append('Simulated: ' + r'$R_b$' + f' = {round(data_rate*100)}%')
        BERs.append(BERs_sim)
        data_Rs.append(data_Rs_sim)
        activation_Rs.append(activation_Rs_sim)

        labels.append('Analytical: ' + r'$R_b$' + f' = {round(data_rate*100)}%')
        BERs.append(BERs_analytical)
        data_Rs.append(data_Rs_analytical)
        activation_Rs.append(activation_Rs_analytical)
    

    su_mimo_svd = SuMimoSVD(Nt=4, Nr=4, c_type='PSK', c_size=4)
        
    BERs_sim, data_Rs_sim, activation_Rs_sim = su_mimo_svd.BERs_simulation(SNRs=SNRs, num_errors=200, num_channels=100)
    BERs_analytical, data_Rs_analytical, activation_Rs_analytical = su_mimo_svd.BERs_analytical(SNRs=SNRs, num_channels=100, settings={'mode': 'approximation', 'eigenchannels': False})
    
    labels.append('Simulated: ' + f'M = {4}%')
    BERs.append(BERs_sim)
    data_Rs.append(data_Rs_sim)
    activation_Rs.append(activation_Rs_sim)

    labels.append('Analytical: ' + f'M = {4}')
    BERs.append(BERs_analytical)
    data_Rs.append(data_Rs_analytical)
    activation_Rs.append(activation_Rs_analytical)


    
    settings = {'labels': labels, 'colors': colors, 'markers': markers, 'opacity': activation_Rs, 'titles': {'BER': f'{str(su_mimo_svd)}' + f' - BER vs SNR', 'data_R': f'{str(su_mimo_svd)}' + f' - Information Bit Rate vs SNR', 'Eb': f'{str(su_mimo_svd)}' + f' - Energy per Bit vs SNR'}}


    su_mimo_svd.plot_performance(SNRs=SNRs, BERs=BERs, data_Rs=data_Rs, settings=settings)
    plt.show()
    