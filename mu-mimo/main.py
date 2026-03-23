# mu-mimo/main.py

import json
from pathlib import Path
import numpy as np
from mu_mimo import *

SIM_CONFIG_PATH = Path(__file__).parent / 'sim_configs.json'
SYSTEM_CONFIG_PATH = Path(__file__).parent / 'system_configs.json'


def _setup_sim_configs(ref_numbers: list[str], filepath: Path) -> dict[str, SimConfig]:
    """
    Set up the simulation configurations for the given reference numbers.

    Parameters
    ----------
    ref_numbers : list[str]
        A list of reference numbers for which to set up the simulation configurations.
    filepath : Path
        The path to the JSON file containing the simulation configurations.
    
    Returns
    -------
    sim_configs : dict[str, SimConfig]
        A dictionary mapping each reference number to its corresponding SimConfig object.
    """
    
    sim_configs = {}
    for ref_number in ref_numbers:
        config_settings = _load_sim_config(filepath, ref_number)
        sim_config = _create_sim_config(config_settings)
        sim_configs[ref_number] = sim_config

    return sim_configs

def _load_sim_config(filepath: Path, ref_number: str) -> dict:
    """
    Load the simulation configuration for a given reference number from a JSON file.

    Parameters
    ----------
    filepath : Path
        The path to the JSON file containing the simulation configurations.
    ref_number : str
        The reference number of the simulation configuration to load.

    Returns
    -------
    config_settings : dict
        The configuration settings for the specified reference number.\\
        The keys of the dictionary correspond to the parameter names, and the values correspond to the effective configuration values.
    """
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for config_settings in data['configurations']:
        if config_settings["Ref. Number"] == ref_number:
            return config_settings

def _create_sim_config(config_settings: dict) -> SimConfig:
    """
    Create a SimConfig object from the given configuration settings.

    Parameters
    ----------
    config_settings : dict
        The configuration settings for the simulation.

    Returns
    -------
    sim_config : SimConfig
        The SimConfig object created from the configuration settings.
    """
    sim_config = SimConfig(
        snr_dB_values               = np.array(config_settings["SNR values (in dB)"], dtype=float),
        num_channel_realizations    = int(config_settings["Channel realizations per SNR value"]),
        num_bit_errors              = int(config_settings["Bit errors per SNR value"]),
        num_bit_errors_scope        = str(config_settings["Scope of bit errors"]),
        M                           = int(config_settings["Transmissions per channel realization"]),
        name                        = "Sim Config " + str(config_settings["Ref. Number"]),
    )

    return sim_config


def _setup_sys_configs(ref_numbers: list[str], filepath: Path) -> dict[str, SystemConfig]:
    """
    Set up the system configurations for the given reference numbers.

    Parameters
    ----------
    ref_numbers : list[str]
        A list of reference numbers for which to set up the system configurations.
    filepath : Path
        The path to the JSON file containing the system configurations.

    Returns
    -------
    system_configs : dict[str, SystemConfig]
        A dictionary mapping each reference number to its corresponding SystemConfig object.
    """
    
    system_configs = {}
    for ref_number in ref_numbers:
        config_settings = _load_sys_config(filepath, ref_number)
        system_config = _create_sys_config(config_settings)
        system_configs[ref_number] = system_config

    return system_configs

def _load_sys_config(filepath: Path, ref_number: str) -> dict:
    """
    Load the system configuration for a given reference number from a JSON file.

    Parameters
    ----------
    filepath : Path
        The path to the JSON file containing the system configurations.
    ref_number : str
        The reference number of the system configuration to load.

    Returns
    -------
    config_settings : dict
        The configuration settings for the specified reference number.\\
        The keys of the dictionary correspond to the parameter names, and the values correspond to the effective configuration values.
    """
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for row in data['configurations']:
        if row[data["configuration_format"]["Ref. Number"]] == ref_number:
            config_settings = {name: row[idx] for name, idx in data['configuration_format'].items()}
            return config_settings
    
    raise ValueError(f"The simulation configuration with ref. number '{ref_number}' not found.")

def _create_sys_config(config_settings: dict) -> SystemConfig:
    """
    Create a SystemConfig object from the given configuration settings.

    Parameters
    ----------
    config_settings : dict
        The configuration settings for the system.

    Returns
    -------
    system_config : SystemConfig
        The SystemConfig object created from the configuration settings.
    """
    
    precoder_mapping = {
        "Neutral": NeutralPrecoder,
        "ZF": ZFPrecoder,
        "BD": BDPrecoder,
        "WMMSE": WMMSEPrecoder,
    }

    combiner_mapping = {
        "Neutral": NeutralCombiner,
        "LSV": LSVCombiner,
    }

    bitloader_mapping = {
        "Neutral": NeutralBitLoader,
        "Fixed": FixedBitLoader,
        "Adaptive": AdaptiveBitLoader,
    }

    mapper_mapping = {
        "Neutral": NeutralMapper,
        "Gray Code": GrayCodeMapper,
    }

    demapper_mapping = {
        "Neutral": NeutralDemapper,
        "Gray Code": GrayCodeDemapper,
    }

    detector_mapping = {
        "Neutral": NeutralDetector,
        "Symbol MD": MDDetector,
    }

    channel_model_mapping = {
        "Neutral": NeutralChannelModel,
        "Rayleigh": IIDRayleighChannelModel,
    }

    noise_model_mapping = {
        "Neutral": NeutralNoiseModel,
        "AWGN": CSAWGNNoiseModel,
    }

    
    # constellation configurations.
    c_configs = ConstConfig(
        types                   = config_settings['Const. Type'],
        sizes                   = int(np.log2(config_settings['Const. Size (fixed)'])) if config_settings['Const. Size (fixed)'] is not None else None,
        capacity_fractions      = config_settings['Const. Size (adaptive)'],
    )

    # base station configurations.
    base_station_configs = BaseStationConfig(
        precoder               = precoder_mapping[config_settings['Precoder']],
        bit_loader             = bitloader_mapping[config_settings['Bit Loader']],
        mapper                 = mapper_mapping[config_settings['Mapper']],
    )

    # channel configurations.
    channel_configs = ChannelConfig(
        channel_model          = channel_model_mapping[config_settings['Channel Model']],
        noise_model            = noise_model_mapping[config_settings['Noise Model']],
    )

    # user terminal configerations.
    user_terminal_configs = UserTerminalConfig(
        combiner              = combiner_mapping[config_settings['Combiner']],
        equalizer             = Equalizer,
        detector              = detector_mapping[config_settings['Detector']],
        demapper              = demapper_mapping[config_settings['Mapper']],
    )


    # system configurations.
    system_config = SystemConfig(
        Pt                    = float(config_settings['Pt']),
        B                     = float(config_settings['B']),
        K                     = int(config_settings['K']),
        Nr                    = int(config_settings['Nr']),
        Nt                    = int(config_settings['Nt']),
        c_configs             = c_configs,
        base_station_configs  = base_station_configs,
        channel_configs       = channel_configs,
        user_terminal_configs = user_terminal_configs,
        name                  = "Ref. System " + str(config_settings['Ref. Number']),
    )

    return system_config



def main(sim_ref_numbers: list[str], sys_ref_numbers: list[str]) -> list[SimResult]:

    results = []

    sim_configs = _setup_sim_configs(sim_ref_numbers, SIM_CONFIG_PATH)
    system_configs = _setup_sys_configs(sys_ref_numbers, SYSTEM_CONFIG_PATH)

    for sim_ref_number in sim_ref_numbers:
        for sys_ref_number in sys_ref_numbers:
            
            runner = SimulationRunner(sim_config=sim_configs[sim_ref_number], system_config=system_configs[sys_ref_number])
            result = runner.run()

            results.append(result)

    return results


if __name__ == "__main__":

    # CHOOSE THE SIMULATION AND SYSTEM CONFIGURATIONS HERE.
    sim_ref_numbers = ["2.1"]
    sys_ref_numbers = [("1.1." + str(i) + "." + str(j)) for i in range(1, 5) for j in range(1, 6)] + [("1.2." + str(i) + "." + str(j)) for i in range(1, 5) for j in range(1, 6)]
    
    # RUN OR LOAD YOUR SIMULATION HERE.
    results = main(sim_ref_numbers, sys_ref_numbers)


    # PLOT YOUR RESULTS HERE.
    for result in results:
        # ResultManager.plot_system_performance(result)
        pass