# In this file, we perform various tests on the SU-MIMO SVD digital communication system.
#
# 0. Basics
#   0.0 Example Simulation
#   0.1 Power and Bit Allocation Plots
#   0.2 Basic Performance Plots
#       0.2.1 Optimal Power Allocation & Adaptive Bit Allocation (R = 100%)
#       0.2.2 Equal Power Allocation & Fixed Bit Allocation (M = 4)
# 
# 1. Simulation Results
#   1.1. Resource Allocation Strategies
#       1.1.0 Scatter Plot
#       1.1.1 Different Data Rates (Optimal Power Allocation)
#       1.1.2 Different Power Allocation Strategies (Optimal vs Equal vs Eigenbeamforming)
#   1.2. Antenna Count
#       1.2.0 Scatter Plot
#       1.2.1 Constant Data Rates
#       1.2.2 Different Data Rates
#   1.3. Eigenchannels
#       1.3.1 Optimal Power Allocation & Adaptive Bit Allocation (R = 100%)
#       1.3.2 Equal Power Allocation & Fixed Bit Allocation (M = 64)
#
# 2. Analytical Results
#   2.1. Basic Performance Plots
#       2.1.1 Optimal Power Allocation & Adaptive Bit Allocation (R = 100%)
#       2.1.2 Equal Power Allocation & Fixed Bit Allocation (M = 4)
#   2.2. Resource Allocation Strategies (Different Data Rates)
#   2.4. Eigenchannels
#       2.4.1 Optimal Power Allocation & Adaptive Bit Allocation (R = 100%)
#       2.4.2 Equal Power Allocation & Fixed Bit Allocation (M = 64)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex
from matplotlib.collections import LineCollection
import itertools
import datetime

from src.su_mimo import SuMimoSVD
from src.transmitter import Transmitter
from src.channel import Channel
from src.receiver import Receiver


### HELPER FUNCTIONS ###

def plot_performance(SNRs, BERs, IBRs, settings):
    """
    Description
    -----------
    Create the BER and/or IBR performance evaluation plots.

    Parameters
    ----------
    SNRs : 1D numpy array (dtype: float, length: N_SNRs)
        The range of signal-to-noise ratio (SNR) values in dB, for which the performance data is provided.
    BERs : 2D numpy array (dtype: float, shape: (N_curves, N_SNRs))
        The simulated Bit Error Rates (BERs) corresponding to each SNR value.
    IBRs : 2D numpy array (dtype: float, shape: (N_curves, N_SNRs))
        The Information Bit Rates (IBRs) (bits per symbol vector) corresponding to each SNR value.
    settings : dict
        A dictionary containing the plot settings.
        - location : the relative location where the plots are saved. (str)
        - labels : the label, for each curve to be displayed in the legend. (list of str, length: N_curves)
        - colors : the color, for each curve. Optional. (list of str, length: N_curves)
        - markers : the type of the markers, for each curve. Optional. (list of str, length: N_curves)
        - marker colors : the color of the markers, for each curve. Optional. (list of str, length: N_curves)
        - opacity : the opacity of the markers and curve parts, for each curve. Optional. (2D numpy array, dtype: float, shape: (N_curves, N_SNRs))
        - titles : the title of each performance evaluation plot. \n
        This also defines which different metrics are plotted! 
        Possible options are:
            * 'BER vs SNR' : Bit Error Rate vs SNR plot.
            * 'IBR vs SNR' : Information Bit Rate vs SNR plot.
            * 'BER vs Eb_N0' : Bit Error Rate vs the ratio of the energy per bit to the noise power spectral density (PSD) plot.
            * 'IBR vs Eb_N0' : Information Bit Rate vs the ratio of the energy per bit to the noise power spectral density (PSD) plot.
    
    Returns
    -------
    plots : dict
        A dictionary containing the created plots. The keys are the plot types, and the values are the corresponding matplotlib figure and axis objects.
    """

    # Initialize a dictionary to hold the result plots.
    plots = {plot: None for plot in settings['titles'].keys()}

    # Initialize the plot settings.
    labels = settings['labels']
    colors = settings.get('colors', [to_hex(c) for c in plt.get_cmap('tab10').colors] + ['black']*(len(labels) - 10))
    markers = settings.get('markers', ['v', '^', '<', '>', 'o', 's', '*', 'd', '8', 'p'] + ['o']*(len(labels) - 10))
    marker_colors = settings.get('marker colors', colors)
    opacity = settings.get('opacity', None)
    titles = settings['titles']

    # Create the requested plots.
    for plot in settings['titles'].keys():

        # Get the data and data specific plot settings.
        if plot == 'BER vs SNR':
            y_data = BERs
            title = titles[plot]
            x_label = 'SNR [dB]'
            y_label = 'Bit Error Rate (BER)'
            y_scale = 'log'
            y_lim_bottom, y_lim_top = 1e-7, 1
        elif plot == 'IBR vs SNR':
            y_data = IBRs
            title = titles[plot]
            x_label = 'SNR [dB]'
            y_label = r'Information Bit Rate ($R_b$) [bits per symbol vector]'
            y_scale = 'linear'
            y_lim_bottom, y_lim_top = None, None
        elif plot == 'BER vs Eb_N0':
            y_data = BERs
            title = titles[plot]
            x_label = r'$E_b/N_0$ [dB]'
            y_label = 'Bit Error Rate (BER)'
            y_scale = 'log'
            y_lim_bottom, y_lim_top = 1e-7, 1
        elif plot == 'IBR vs Eb_N0':
            y_data = IBRs
            title = titles[plot]
            x_label = r'$E_b/N_0$ [dB]'
            y_label = r'Information Bit Rate ($R_b$) [bits per symbol vector]'
            y_scale = 'linear'
            y_lim_bottom, y_lim_top = None, None
        else: 
            raise ValueError(f'The performance evaluation plot "{plot}" is not recognized.')

        # Initialize the plot.
        fig, ax = plt.subplots()

        # Create each curve.
        for i in range(len(labels)):

            x_data_i = (SNRs if plot[-3:] == 'SNR' else (SNRs - np.where(IBRs[i] == 0, np.nan, 10*np.log10(IBRs[i]))))[~np.isnan(y_data[i])]
            y_data_i = y_data[i][~np.isnan(y_data[i])]
            if opacity[i] is not None: opacity_i = opacity[i][~np.isnan(y_data[i])]
            
            if opacity[i] is not None:
                
                points = np.array([x_data_i, y_data_i]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                segment_colors = np.tile(to_rgba(colors[i]), (len(segments), 1))
                segment_colors[:, 3] = (opacity_i[:-1] + opacity_i[1:]) / 2
                
                lc = LineCollection(segments, colors=segment_colors, linewidth=1.5)
                ax.add_collection(lc)

                for j in range(len(x_data_i)):
                    ax.scatter(x_data_i[j], y_data_i[j], marker=markers[i], color=marker_colors[i], edgecolors=marker_colors[i], alpha=opacity_i[j], s=36, zorder=3)

                ax.plot([], [], label=(labels[i] if i < len(labels) else None), color=colors[i], marker=markers[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], linestyle='-')
            
            else:
                ax.plot(x_data_i, y_data_i, label=(labels[i] if i < len(labels) else None), color=colors[i], marker=markers[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], linestyle='-', alpha =1.0)
        
        # Set the plot settings.
        ax.set_title(title)
        ax.set_yscale(y_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_lim_bottom, y_lim_top)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()
        fig.tight_layout()
        
        # Store the plot.
        location = 'su-mimo/report/plots/' + settings.get('location', '')
        filename = f'{plot.replace(" ", "_")}.png'
        fig.savefig(location + filename, dpi=300, bbox_inches='tight')
        plots[plot] = (fig, ax)
    
    return plots

def load_results(location, filename, SNRs):

    print(f"Loading existing data ...\nfilename: {filename}\n")
    data = np.load(location + filename, allow_pickle=True)

    BERs = np.full_like(SNRs, np.nan, dtype=float)
    IBRs = np.full_like(SNRs, np.nan, dtype=float)
    ARs  = np.full_like(SNRs, np.nan, dtype=float)

    mask = np.isin(SNRs, data['SNRs'])
    idx  = np.searchsorted(data['SNRs'], SNRs[mask])

    BERs[mask] = data['BERs'][idx]
    IBRs[mask] = data['IBRs'][idx]
    ARs[mask]  = data['ARs'][idx]

    return BERs, IBRs, ARs

def update_results(SNRs_new, Nt, Nr, c_type, RAS, mode, eigenchannels=False, printing=True):

    # Load old data.
    location = 'su-mimo/report/plots/' + ('simulation_results/' if mode == 'simulation' else 'analytical_results/' + ('approximations/' if mode == 'approximation' else 'upper_bounds/'))
    filename = f"{Nt}x{Nr}_{c_type}__pa_{RAS['power allocation']}__" + (f"ba_adaptive__R__{RAS['data rate']}" if RAS['bit allocation'] == 'adaptive' else f"ba_fixed__M__{RAS['constellation sizes']}") + ".npz"
    data = np.load(location + filename, allow_pickle=True)
    
    SNRs_old = data['SNRs']
    BERs_old = data['BERs']
    IBRs_old = data['IBRs']
    ARs_old = data['ARs']

    if printing:
        print("Loaded files: ")
        print(f"  Old SNRs: {SNRs_old}")
        print(f"  Old BERs: {np.array2string(BERs_old, formatter={'float_kind': lambda x: f"{x:.2e}"})}")
        print(f"  Old IBRs: {np.array2string(IBRs_old, formatter={'float_kind': lambda x: f"{x:.2e}"})}")
        print(f"  Old ARs: {np.array2string(ARs_old, formatter={'float_kind': lambda x: f"{x:.2e}"})}")
    
    # Simulate for new SNRs.
    su_mimo_svd = SuMimoSVD(Nt=Nt, Nr=Nr, c_type=c_type, Pt=1.0, B=0.5, RAS=RAS, SNR=SNRs_new[0], H=None)
    if mode == 'simulation':
        if not eigenchannels: BERs_new, IBRs_new, ARs_new = su_mimo_svd.BERs_simulation(SNRs=SNRs_new, num_channels=1000)
        else: BERs_new, IBRs_new, ARs_new = su_mimo_svd.BERs_eigenchs_simulation(SNRs=SNRs_new, num_channels=1000)
    else:
        BERs_new, IBRs_new, ARs_new = su_mimo_svd.BERs_analytical(SNRs=SNRs_new, num_channels=1000, settings={'mode': mode, 'eigenchannels': eigenchannels})


    # Combine old and new data.
    SNRs = np.union1d(SNRs_old, SNRs_new)

    BERs = np.full_like(SNRs, np.nan, dtype=float)
    IBRs = np.full_like(SNRs, np.nan, dtype=float)
    ARs  = np.full_like(SNRs, np.nan, dtype=float)

    mask_old = np.isin(SNRs, SNRs_old)
    idx_old  = np.searchsorted(SNRs_old, SNRs[mask_old])
    BERs[mask_old] = BERs_old[idx_old]
    IBRs[mask_old] = IBRs_old[idx_old]
    ARs[mask_old]  = ARs_old[idx_old]

    mask_new = np.isin(SNRs, SNRs_new)
    idx_new  = np.searchsorted(SNRs_new, SNRs[mask_new])
    BERs[mask_new] = BERs_new[idx_new]
    IBRs[mask_new] = IBRs_new[idx_new]
    ARs[mask_new]  = ARs_new[idx_new]

    if printing:
        print("Updated files: ")
        print(f"  New SNRs: {SNRs}")
        print(f"  New BERs: {np.array2string(BERs, formatter={'float_kind': lambda x: f"{x:.2e}"})}")
        print(f"  New IBRs: {np.array2string(IBRs, formatter={'float_kind': lambda x: f"{x:.2e}"})}")
        print(f"  New ARs: {np.array2string(ARs, formatter={'float_kind': lambda x: f"{x:.2e}"})}")


    # Save updated data.
    np.savez(filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)
    return


### TESTS SETUP ###

def test_0_0(Nt, Nr, c_type, Pt=1.0, B=0.5, RAS=None, SNR=None, H=None):

    if RAS is None: RAS = {'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': 1.0, 'constellation sizes': None, 'control channel': True}
    if SNR is None: SNR = 10
    if H is None: H = np.array([[3, 2, 2], [2, 3, -2]])
    
    su_mimo_svd = SuMimoSVD(Nt=Nt, Nr=Nr, c_type=c_type, Pt=Pt, B=B, RAS=RAS, SNR=SNR, H=H)
    ber = su_mimo_svd.print_simulation_example(bitstream=np.random.randint(0, 2, size=2000), K=Nt)
    
    return ber

def test_0_1(Nt_values, Nr_values, c_type_values, Pt_values=(1.0,), RASs=(None,), SNRs=(12,)):

    for Nt, Nr, c_type, Pt, RAS, SNR in itertools.product(Nt_values, Nr_values, c_type_values, Pt_values, RASs, SNRs):

        if RAS is None: RAS = {'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': 1.0, 'constellation sizes': None, 'control channel': True}
        
        transmitter = Transmitter(Nt=Nt, c_type=c_type, Pt=Pt, B=0.5, RAS=RAS)
        channel = Channel(Nt=Nt, Nr=Nr, SNR=SNR)

        transmitter.plot_power_allocation(CSIT=channel.get_CSI())
        transmitter.plot_bit_allocation(CSIT=channel.get_CSI())
    
    return

def test_0_2_1(Nt, Nr, c_types, SNRs):

    title = f"{Nt}x{Nr} SU-MIMO SVD DigCom System\n" + "Optimal Power Allocation & Adaptive Bit Allocation (R = 100%)"
    BERs_list = []
    IBRs_list = []
    settings = {
        'location': '1_simulation/0_basics/2_1__' + f'{Nt}x{Nr}_' + '_'.join(c_types) + '__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'labels': ['PAM', 'PSK', 'QAM'],
        'colors': ['tab:blue']*3,
        'markers': ['|', 'o', 's'],
        'opacity': [],
    }


    for c_type in c_types:
        
        # Initialization.
        location = "su-mimo/report/plots/simulation_results/"
        filename = f"{Nt}x{Nr}_{c_type}" + f"__pa_optimal__ba_adaptive__R_100" + ".npz"

        # Simulation.
        if os.path.exists(location + filename):
            BERs, IBRs, ARs = load_results(location, filename, SNRs)
        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type)
            BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs)
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

        # Store the results.
        BERs_list.append(BERs)
        IBRs_list.append(IBRs)
        settings['opacity'].append(ARs)


    plots = plot_performance(SNRs=SNRs, BERs=BERs_list, IBRs=IBRs_list, settings=settings)
    return plots

def test_0_2_2(Nt, Nr, c_types, SNRs):
    
    title = f'{Nt}x{Nr} SU-MIMO SVD DigCom System\n' + 'Equal Power Allocation & Fixed Bit Allocation (M = 4)'
    BERs_list = []
    IBRs_list = []
    settings = {
        'location': '1_simulation/0_basics/2_2__' + f'{Nt}x{Nr}_' + '_'.join(c_types) + '__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'labels': ['PAM', 'PSK', 'QAM'],
        'colors': ['tab:green']*3,
        'markers': ['|', 'o', 's'],
        'opacity': [],
    }

    for c_type in c_types:
            
        # Initialization.
        location = 'su-mimo/report/plots/simulation_results/'
        filename = f'{Nt}x{Nr}_{c_type}' + f'__pa_equal__ba_fixed__M_4' + '.npz'

        # Simulation.
        if os.path.exists(location + filename): 
            BERs, IBRs, ARs = load_results(location, filename, SNRs)
        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type)
            su_mimo_svd.set_RAS({'power allocation': 'equal', 'bit allocation': 'fixed', 'data rate': None, 'constellation sizes': 2**4, 'control channel': True}) 
            
            BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs)
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

        # Termination.
        BERs_list.append(BERs)
        IBRs_list.append(IBRs)
        settings['opacity'].append(ARs)

    plots = plot_performance(SNRs=SNRs, BERs=BERs_list, IBRs=IBRs_list, settings=settings)
    return plots


def test_1_1_0(Nt, Nr, c_type, SNR, data_rates):

    su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, SNR=SNR)

    plots = []
    for data_rate in data_rates:

        su_mimo_svd.set_RAS({'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': data_rate, 'constellation sizes': None, 'control channel': True})
        plot = su_mimo_svd.plot_scatter_diagram(K=5000)
        plots.append(plot)

    return plots

def test_1_1_1(Nt, Nr, c_type, SNRs, data_rates):

    su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, SNR=SNRs[0])
    title = 'RAS Performance Comparison\n' + str(su_mimo_svd)

    BERs_list = []
    IBRs_list = []
    settings = {
        'location': '1_simulation/1_resource_allocation_strategies/data_rates/' + f'1__{Nt}x{Nr}_{c_type}__R' + '_'.join([str(R) for R in data_rates]) + '__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['tab:blue', 'cornflowerblue', 'skyblue', 'lightskyblue'],
        'labels': [r'$R \approx $' + f'{R/100:.0%}' for R in data_rates],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*3,
        'opacity': [],
    }

    for i, R in enumerate(data_rates):
        
        # Initialization.
        location = 'su-mimo/report/plots/simulation_results/'
        filename = f'{Nt}x{Nr}_{c_type}' + f'__pa_optimal__ba_adaptive__R_{R}' + '.npz'
        
        # Simulation.
        if os.path.exists(location + filename):
            BERs, IBRs, ARs = load_results(location, filename, SNRs)
        
        else:
            su_mimo_svd.set_RAS({'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': R/100, 'constellation sizes': None, 'control channel': True})
            BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs[i:])
            
            BERs = np.concatenate( (np.full(i, np.nan), BERs) )
            IBRs = np.concatenate( (np.full(i, np.nan), IBRs) )
            ARs = np.concatenate( (np.full(i, np.nan), ARs) )
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)
        
        # Store the results.
        BERs_list.append(BERs)
        IBRs_list.append(IBRs)
        settings['opacity'].append(ARs)

    plots = plot_performance(SNRs=SNRs, BERs=BERs_list, IBRs=IBRs_list, settings=settings)
    return plots

def test_1_1_2(Nt, Nr, c_type, SNRs):

    # Initialization.
    su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, SNR=0)
    title = 'RAS Performance Comparison\n' + str(su_mimo_svd)
    RAS = su_mimo_svd.RAS

    BERs_curves = []
    IBRs_curves = []
    settings = {
        'location': '1_simulation/1_resource_allocation_strategies/power_allocation/' + f'2__{Nt}x{Nr}_{c_type}__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['tab:blue', 'tab:green', 'tab:olive'],
        'labels': ['optimal power allocation', 'equal power allocation', 'eigenbeamforming'],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*3,
        'opacity': [],
    }

    
    # 1. Fisrt curve. PA: optimal, BA: adaptive, R=100% (default blue curve).
    location = 'su-mimo/report/plots/simulation_results/'
    filename = f'{Nt}x{Nr}_{c_type}__pa_optimal__ba_adaptive__R_100.npz'
    
    if os.path.exists(location + filename): BERs, IBRs, ARs = load_results(location, filename, SNRs)
    else: BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs)

    BERs_curves.append(BERs)
    IBRs_curves.append(IBRs)
    settings['opacity'].append(ARs)

    # 2. Second and third curve. PA: equal and eigenbeamforming, BA: fixed, M chosen to match IBRs of first curve.
    for pas in ['equal', 'eigenbeamforming']:

        location = 'su-mimo/report/plots/simulation_results/'
        filename = f'{Nt}x{Nr}_{c_type}__pa_{pas}__ba_fixed__M_variable.npz'

        if os.path.exists(location + filename):
            BERs, IBRs, ARs = load_results(location, filename, SNRs)

        else:  
            RAS |= {'power allocation': pas, 'bit allocation': 'fixed'}
            
            BERs = np.full_like(SNRs, np.nan, dtype=float)
            IBRs = np.full_like(SNRs, np.nan, dtype=float)
            ARs = np.full_like(SNRs, np.nan, dtype=float)

            for SNR, IBR in zip(SNRs, IBRs_curves['optimal']):
                
                mc = (IBR / (su_mimo_svd.Nt if pas == 'equal' else 1))
                mc = 2*round( mc / 2 ) if c_type == 'QAM' else round( mc )
                mc = max(mc, (2 if c_type == 'QAM' else 1))

                RAS |= {'constellation sizes': 2**(mc)}
                su_mimo_svd.set_RAS(RAS)

                print(f"\n\nSimulating for SNR: {SNR} dB, PA: {pas}, target IBR: {round(IBR, 1)}, IBR: {mc} ...")
                BER, IBR, AR = su_mimo_svd.BERs_simulation(SNRs=[SNR])

                BERs[SNRs == SNR] = BER[0]
                IBRs[SNRs == SNR] = IBR[0]
                ARs[SNRs == SNR] = AR[0]

            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)
        
        BERs_curves.append(BERs)
        IBRs_curves.append(IBRs)
        settings['opacity'].append(ARs)

    # 3. Plotting.
    plots = plot_performance(SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, settings=settings)
    return plots

def test_1_2_0(c_type, SNR, antenna_counts):
    
    plots = [SuMimoSVD(Nt, Nr, c_type, SNR=SNR).plot_scatter_diagram(K=5000) for Nt, Nr in antenna_counts]
    return plots

def test_1_2_1(c_type, SNRs, antenna_counts):

    # 1. Initialization.
    title = 'Antenna Count Performance Comparison\n' + f'{c_type} SU-MIMO SVD DigCom System'
    BERs_curves = []
    IBRs_curves = []
    settings = {
        'location': '1_simulation/2_antenna_count/1__' + '_'.join([f'{Nt}x{Nr}' for Nt, Nr in antenna_counts]) + f'_{c_type}__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['tab:orange', 'tab:blue', 'tab:green', 'tab:olive', 'tab:red'],
        'labels': [f'{Nt}x{Nr} Antennas' for Nt, Nr in antenna_counts],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*len(antenna_counts),
        'opacity': [],
    } 

    # 2. Simulation for different antenna counts.
    for Nt, Nr in antenna_counts:
        
        # Initialization.
        location = 'su-mimo/report/plots/simulation_results/'
        filename = f'{Nt}x{Nr}_{c_type}__pa_optimal__ba_adaptive__R_100.npz'

        # Simulation.
        if os.path.exists(location + filename):
            BERs, IBRs, ARs = load_results(location, filename, SNRs)
        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type)
            BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs)
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

        # Store the results.
        BERs_curves.append(BERs)
        IBRs_curves.append(IBRs)
        settings['opacity'].append(ARs)
    

    # 3. Plotting.
    plots = plot_performance(SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, settings=settings)
    return plots

def test_1_2_2(c_type, SNRs, antenna_counts_data_rates):
    
    # 1. Initialization.
    title = 'Antenna Count Performance Comparison\n' + f'{c_type} SU-MIMO SVD DigCom System'
    BERs_curves = []
    IBRs_curves = []
    settings = {
        'location': '1_simulation/2_antenna_count/2__' + '_'.join([f'{Nt}x{Nr}_R_{data_rate}' for Nt, Nr, data_rate in antenna_counts_data_rates]) + f'_{c_type}__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['navy', 'tab:blue', 'cornflowerblue', 'skyblue', 'lightskyblue'],
        'labels': [f'{Nt}x{Nr} Antennas' + r', $R \approx $'+f'{(data_rate/100):.0%}' for Nt, Nr, data_rate in antenna_counts_data_rates],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*len(antenna_counts_data_rates),
        'opacity': [],
    }

    # 2. Simulation for different antenna counts.
    for Nt, Nr, data_rate in antenna_counts_data_rates:
        
        # Initialization.
        location = 'su-mimo/report/plots/simulation_results/'
        filename = f'{Nt}x{Nr}_{c_type}__pa_optimal__ba_adaptive__R_{data_rate:.0f}.npz'

        # Simulation.
        if os.path.exists(location + filename):
            BERs, IBRs, ARs = load_results(location, filename, SNRs)
        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, RAS={'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': data_rate/100, 'constellation sizes': None, 'control channel': True})
            BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs)
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

        # Store the results.
        BERs_curves.append(BERs)
        IBRs_curves.append(IBRs)
        settings['opacity'].append(ARs)
    
    # 3. Plotting.
    plots = plot_performance(SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, settings=settings)
    return plots

def test_1_3_1(Nt, Nr, c_type, SNRs):

    # 1. Initialization.
    su_mimo_svd = SuMimoSVD(Nt, Nr, c_type)
    title = 'Eigenchannel Performance Comparison\n' + str(su_mimo_svd)
    settings = {
        'location': '1_simulation/3_eigenchannels/1__' + f'{Nt}x{Nr}_{c_type}__R_100__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['midnightblue', 'mediumblue', 'dodgerblue', 'lightskyblue'],
        'labels': ['Eigenchannel ' + str(i+1) for i in range(min(Nt, Nr))],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*min(Nt, Nr),
        'opacity': [],
    }

    # 2. Simulation for different antenna counts.
    location = 'su-mimo/report/plots/simulation_results/'
    filename = f'{Nt}x{Nr}_{c_type}__pa_optimal__ba_adaptive__R_100__eigenchannels.npz'

    if os.path.exists(location + filename):
        
        print(f"Loading existing data  ...\nfilename: {location + filename}\n")
        data = np.load(location + filename, allow_pickle=True)

        BERs_curves = np.full((min(su_mimo_svd.Nt, su_mimo_svd.Nr), len(SNRs)), np.nan, dtype=float)
        IBRs_curves = np.full((min(su_mimo_svd.Nt, su_mimo_svd.Nr), len(SNRs)), np.nan, dtype=float)
        ARs_curves  = np.full((min(su_mimo_svd.Nt, su_mimo_svd.Nr), len(SNRs)), np.nan, dtype=float)

        mask = np.isin(SNRs, data['SNRs'])
        idx  = np.searchsorted(data['SNRs'], SNRs[mask])

        BERs_curves[:, mask] = data['BERs'][ :, idx ]
        IBRs_curves[:, mask] = data['IBRs'][ :, idx ]
        ARs_curves[:, mask]  = data['ARs'][  :, idx ]
        settings['opacity'] = [ARs_curves[s] for s in range(ARs_curves.shape[0])]
    
    else:
        BERs_curves, IBRs_curves, ARs_curves = su_mimo_svd.BERs_eigenchs_simulation(SNRs=SNRs)
        settings['opacity'] = ARs_curves.tolist()
        np.savez(location + filename, SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, ARs=ARs_curves)
    
    # 3. Plotting.
    plots = plot_performance(SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, settings=settings)
    return plots

def test_1_3_2(Nt, Nr, c_type, SNRs, mc):

    # 1. Initialization.
    su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, RAS={'power allocation': 'equal', 'bit allocation': 'fixed', 'data rate': None, 'constellation sizes': 2**mc, 'control channel': True})
    title = 'Eigenchannel Performance Comparison\n' + str(su_mimo_svd)
    settings = {
        'location': '1_simulation/3_eigenchannels/2__' + f'{Nt}x{Nr}_{c_type}__M_{mc}__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['darkgreen', 'forestgreen', 'yellowgreen', 'greenyellow'],
        'labels': ['Eigenchannel ' + str(i+1) for i in range(min(Nt, Nr))],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*min(Nt, Nr),
        'opacity': [None]*min(Nt, Nr),
    }

    # 2. Simulation for different antenna counts.
    location = 'su-mimo/report/plots/simulation_results/'
    filename = f'{Nt}x{Nr}_{c_type}__pa_equal__ba_fixed__M_{mc}__eigenchannels.npz'

    if os.path.exists(location + filename):
        
        print(f"Loading existing data  ... \nfilename: {location + filename}\n")
        data = np.load(location + filename, allow_pickle=True)

        BERs_curves = np.full((min(su_mimo_svd.Nt, su_mimo_svd.Nr), len(SNRs)), np.nan, dtype=float)
        IBRs_curves = np.full((min(su_mimo_svd.Nt, su_mimo_svd.Nr), len(SNRs)), np.nan, dtype=float)

        mask = np.isin(SNRs, data['SNRs'])
        idx  = np.searchsorted(data['SNRs'], SNRs[mask])

        BERs_curves[:, mask] = data['BERs'][ :, idx ]
        IBRs_curves[:, mask] = data['IBRs'][ :, idx ]
    
    else:
        BERs_curves, IBRs_curves, _ = su_mimo_svd.BERs_eigenchs_simulation(SNRs=SNRs)
        np.savez(location + filename, SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves)
    
    # 3. Plotting.
    plots = plot_performance(SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, settings=settings)
    return plots


def test_2_0_1(Nt, Nr, c_type, SNRs, modes):

    title = f"{Nt}x{Nr} {c_type} SU-MIMO SVD DigCom System\n" + "Optimal Power Allocation & Adaptive Bit Allocation (R = 100%)"
    BERs_list = []
    IBRs_list = []
    settings = {
        'location': '2__analytical/0_basics/1__' + f'{Nt}x{Nr}_{c_type}' + ('__'.join(modes)).replace(' ', '_') + '__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'labels': modes,
        'colors': ['tab:blue', 'tab:orange', 'tab:red'],
        'markers': ['s']*3,
        'opacity': [],
    }

    for mode in modes:

        # Initialization.
        location = "su-mimo/report/plots/simulation_results/" if mode == 'simulation' else "su-mimo/report/plots/analytical_results/" + ("approximations/" if mode == 'approximation' else "upper_bounds/")
        filename = f"{Nt}x{Nr}_{c_type}" + f"__pa_optimal__ba_adaptive__R_100" + ".npz"

        # Simulation.
        if os.path.exists(location + filename):
            BERs, IBRs, ARs = load_results(location, filename, SNRs)
        
        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type)
            if mode == 'simulation': BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs)
            else: BERs, IBRs, ARs = su_mimo_svd.BERs_analytical(SNRs=SNRs, num_channels=1000, settings={'mode': mode, 'eigenchannels': False})
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

        # Store the results.
        BERs_list.append(BERs)
        IBRs_list.append(IBRs)
        settings['opacity'].append(ARs)


    plots = plot_performance(SNRs=SNRs, BERs=BERs_list, IBRs=IBRs_list, settings=settings)
    return plots

def test_2_0_2(Nt, Nr, c_type, SNRs, modes):
    
    title = f'{Nt}x{Nr} {c_type} SU-MIMO SVD DigCom System\n' + 'Equal Power Allocation & Fixed Bit Allocation (4 bits per antenna)'
    BERs_list = []
    IBRs_list = []
    settings = {
        'location': '2__analytical/0_basics/2__' + f'{Nt}x{Nr}_{c_type}__' + ('__'.join(modes)).replace(' ', '_') + '__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'labels': modes,
        'colors': ['tab:green', 'tab:orange', 'tab:red'],
        'markers': ['s']*3,
        'opacity': [],
    }


    for mode in modes:
        
        # Initialization.
        location = 'su-mimo/report/plots/simulation_results/' if mode == 'simulation' else 'su-mimo/report/plots/analytical_results/' + (mode.replace(' ', '_') + 's/')
        filename = f'{Nt}x{Nr}_{c_type}' + f'__pa_equal__ba_fixed__M_4' + '.npz'

        # Simulation.
        if os.path.exists(location + filename):
            BERs, IBRs, ARs = load_results(location, filename, SNRs)
        
        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type)
            su_mimo_svd.set_RAS({'power allocation': 'equal', 'bit allocation': 'fixed', 'data rate': None, 'constellation sizes': 2**4, 'control channel': True})
            
            if mode == 'simulation': BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs)
            else: BERs, IBRs, ARs = su_mimo_svd.BERs_analytical(SNRs=SNRs, num_channels=500, settings={'mode': mode, 'eigenchannels': False})
            
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

        # Store the results.
        BERs_list.append(BERs)
        IBRs_list.append(IBRs)
        settings['opacity'].append(ARs)


    plots = plot_performance(SNRs=SNRs, BERs=BERs_list, IBRs=IBRs_list, settings=settings)
    return plots

def test_2_1(Nt, Nr, c_type, SNRs, modes, data_rates):

    title = 'RAS Performance Comparison\n' + f'{Nt}x{Nr} {c_type} SU-MIMO SVD DigCom System'
    BERs_list = []
    IBRs_list = []
    settings = {
        'location': '2_analytical/1_resource_allocation_strategies/1__' + f'{Nt}x{Nr}_{c_type}__R_' + '_'.join([str(data_rate) for data_rate in data_rates]) + '__' + ('__'.join(modes)).replace(' ', '_') + '__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['tab:blue', 'cornflowerblue', 'lightskyblue'] + ['tab:orange', 'orange', 'gold'] + ['darkred', 'tab:red', 'orangered'],
        'labels': [r'$R \approx $' + f'{data_rate:.0%}' + ' (' + mode + ')' for mode in modes for data_rate in data_rates],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*(len(data_rates)*len(modes)),
        'opacity': [],
    }

    for mode in modes:

        for i, data_rate in enumerate(data_rates):
            
            # Initialization.
            location = 'su-mimo/report/plots/' + ('simulation_results/' if mode == 'simulation' else 'analytical_results/' + (mode.replace(' ', '_') + 's/'))
            filename = f'{Nt}x{Nr}_{c_type}__pa_optimal__ba_adaptive__R_{data_rate}.npz'
            
            # Simulation.
            if os.path.exists(location + filename):
                BERs, IBRs, ARs = load_results(location, filename, SNRs)

            else:
                su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, SNR=SNRs[0])
                su_mimo_svd.set_RAS({'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': data_rate/100, 'constellation sizes': None, 'control channel': True})
                
                if mode == 'simulation':
                    BERs, IBRs, ARs = su_mimo_svd.BERs_simulation(SNRs=SNRs[i:])
                    BERs = np.concatenate( (np.full(i, np.nan), BERs) )
                    IBRs = np.concatenate( (np.full(i, np.nan), IBRs) )
                    ARs  = np.concatenate( (np.full(i, np.nan), ARs) )
            
                else:
                    BERs, IBRs, ARs = su_mimo_svd.BERs_analytical(SNRs=SNRs, num_channels=10000, settings={'mode': mode, 'eigenchannels': False})
                
                np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

            # Store the results.
            BERs_list.append(BERs)
            IBRs_list.append(IBRs)
            settings['opacity'].append(ARs)

    plots = plot_performance(SNRs=SNRs, BERs=BERs_list, IBRs=IBRs_list, settings=settings)
    return plots

def test_2_2_1(Nt, Nr, c_type, SNRs, modes):

    title = 'Eigenchannel Performance Comparison\n' + f'{Nt}x{Nr} {c_type} SU-MIMO SVD DigCom System'
    BERs_curves = []
    IBRs_curves = []
    settings = {
        'location': '2_analytical/2_eigenchannels/1__' + f'{Nt}x{Nr}_{c_type}__' + ('__'.join(modes)).replace(' ', '_') + '__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['mediumblue', 'dodgerblue'] + ['darkorange', 'gold'],
        'labels': ['Eigenchannel ' + str(i+1) + ' (' + mode + ')' for mode in modes for i in range(2)],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*len(modes)*2,
        'opacity': [],
    }


    for mode in modes:

        # Initialization.
        location = 'su-mimo/report/plots/' + ('simulation_results/' if mode == 'simulation' else ('analytical_results/' + mode.replace(' ', '_') + 's/'))
        filename = f'{Nt}x{Nr}_{c_type}__pa_optimal__ba_adaptive__R_100__eigenchannels.npz'
        
        # Simulation.
        if os.path.exists(location + filename):
            
            print(f"Loading existing data  ...\nfilename: {location + filename}\n")
            data = np.load(location + filename, allow_pickle=True)

            BERs = np.full((min(Nt, Nr), len(SNRs)), np.nan, dtype=float)
            IBRs = np.full((min(Nt, Nr), len(SNRs)), np.nan, dtype=float)
            ARs  = np.full((min(Nt, Nr), len(SNRs)), np.nan, dtype=float)

            mask = np.isin(SNRs, data['SNRs'])
            idx  = np.searchsorted(data['SNRs'], SNRs[mask])

            BERs[:, mask] = data['BERs'][:, idx]
            IBRs[:, mask] = data['IBRs'][:, idx]
            ARs[:, mask]  = data['ARs'][:, idx]

        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type)
            su_mimo_svd.set_RAS({'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': 1.0, 'constellation sizes': None, 'control channel': True})
            
            if mode == 'simulation': BERs, IBRs, ARs = su_mimo_svd.BERs_eigenchs_simulation(SNRs=SNRs, num_channels=500)
            else: BERs, IBRs, ARs = su_mimo_svd.BERs_analytical(SNRs=SNRs, num_channels=1000, settings={'mode': mode, 'eigenchannels': True})
            
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs, ARs=ARs)

        # Store the results.
        BERs_curves.extend(BERs.tolist())
        IBRs_curves.extend(IBRs.tolist())
        settings['opacity'].extend(ARs.tolist())
    

    plots = plot_performance(SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, settings=settings)
    return plots

def test_2_2_2(Nt, Nr, c_type, SNRs, modes, mc):

    title = 'Eigenchannel Performance Comparison\n' + f'{Nt}x{Nr} {c_type} SU-MIMO SVD DigCom System'
    BERs_curves = []
    IBRs_curves = []
    settings = {
        'location': '2_analytical/2_eigenchannels/2__' + f'{Nt}x{Nr}_{c_type}__' + ('__'.join(modes)).replace(' ', '_') + f'__M_{mc}__',
        'titles': {'BER vs SNR': title, 'IBR vs SNR': title, 'BER vs Eb_N0': title, 'IBR vs Eb_N0': title},
        'colors': ['forestgreen', 'yellowgreen'] + ['darkorange', 'orange'], 
        'labels': ['Eigenchannel ' + str(i+1) + ' (' + mode + ')' for mode in modes for i in range(min(Nt, Nr))],
        'markers': ['s' if c_type == 'QAM' else 'o' if c_type == 'PSK' else '|']*(min(Nt, Nr)*len(modes)),
        'opacity': [None]*(min(Nt, Nr)*len(modes)),
    }

    for mode in modes:
        
        location = 'su-mimo/report/plots/' + ('simulation_results/' if mode == 'simulation' else ('analytical_results/' + mode.replace(' ', '_') + 's/'))
        filename = f'{Nt}x{Nr}_{c_type}__pa_equal__ba_fixed__M_{mc}__eigenchannels.npz'

        if os.path.exists(location + filename):
            
            print(f"Loading existing data  ...\nfilename: {location + filename}\n")
            data = np.load(location + filename, allow_pickle=True)

            BERs = np.full((min(Nt, Nr), len(SNRs)), np.nan, dtype=float)
            IBRs = np.full((min(Nt, Nr), len(SNRs)), np.nan, dtype=float)
    
            mask = np.isin(SNRs, data['SNRs'])
            idx  = np.searchsorted(data['SNRs'], SNRs[mask])

            BERs[:, mask] = data['BERs'][:, idx]
            IBRs[:, mask] = data['IBRs'][:, idx]
        
        else:
            su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, RAS={'power allocation': 'equal', 'bit allocation': 'fixed', 'data rate': None, 'constellation sizes': 2**mc, 'control channel': True})
        
            if mode == 'simulation': BERs, IBRs, _ = su_mimo_svd.BERs_eigenchs_simulation(SNRs=SNRs, num_channels=1000)
            else: BERs, IBRs, _ = su_mimo_svd.BERs_analytical(SNRs=SNRs, num_channels=1000, settings={'mode': mode, 'eigenchannels': True})
            np.savez(location + filename, SNRs=SNRs, BERs=BERs, IBRs=IBRs)
        
        BERs_curves.extend(BERs)
        IBRs_curves.extend(IBRs)
        
    plots = plot_performance(SNRs=SNRs, BERs=BERs_curves, IBRs=IBRs_curves, settings=settings)
    return plots


### TESTS ###

# example:
# plots = test_0_2_1(Nt=4, Nr=4, c_types=['PAM', 'PSK', 'QAM'], SNRs=np.arange(0, 31, 5))
# plt.show()
for i in range(10):
    test_1_2_0('QAM', 15, [(2, 2)])