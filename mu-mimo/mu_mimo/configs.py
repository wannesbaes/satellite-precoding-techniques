# mu-mimo/mu_mimo/configs.py

from dataclasses import dataclass, field
import numpy as np
from typing import Literal
import json
from pathlib import Path

from .processing import *
from .types import IntArray, RealArray, ConstType


@dataclass()
class ConstConfig:
    """
    The constellation configuration settings for the data transmission between the BS and one UT.

    Parameters
    ----------
    types : list[ConstType], shape (K,)
        The constellation types for the data streams to each UT. \\
        (If the same constellation type is used for all UTs, this can also be provided as a single ConstType at the moment of initialization.)
    sizes : IntArray, shape (K,) | None
        The constellation sizes in bits, i.e. the number of bits per data symbol (point in the constellation), for the data streams to each UT. \\
        If the same constellation size is used for all UTs, this can also be provided as a single int at the moment of initialization. \n
        In case of adaptive bit allocation, the constellation sizes are not predetermined but will be calculated by the bit allocator. In that case, this can be set to None.
    capacity_fractions : RealArray, shape (K,) | None
        The fractions of channel capacities that are allocated to each UT. \\
        If the same capacity fraction is allocated to all UTs, this can also be provided as a single float at the moment of initialization. \n
        For adaptive bit allocation, the bit allocator computes the achievable rates (shannon capacity) for each stream of all UTs. Then it calculates the information bit rates for the data streams to each UT as the fraction of their achievable rates.\\
        In case of fixed bit allocation, the information bit rates are predetermined. In that case, this can be set to None.
    """
    types: ConstType | list[ConstType]
    sizes: int | IntArray | None = None
    capacity_fractions: float | RealArray | None = None

    def __str__(self) -> str:
        
        lines = []

        if len(set(self.types)) == 1:
            lines.append(f"type: {self.types[0]}")
        else:
            for i, c_type in enumerate(self.types):
                lines.append(f"type_{i}: {c_type}")
        
        if self.sizes is not None:
            if np.all(self.sizes == self.sizes[0]):
                lines.append(f"size: {self.sizes[0]}")
            else:
                for i, size in enumerate(self.sizes):
                    lines.append(f"size_{i}: {size}")

        if self.capacity_fractions is not None:
            if np.all(self.capacity_fractions == self.capacity_fractions[0]):
                lines.append(f"capacity_fraction: {self.capacity_fractions[0]}")
            else:
                for i, fraction in enumerate(self.capacity_fractions):
                    lines.append(f"capacity_fraction_{i}: {fraction}")

        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, ConstConfig):
            return NotImplemented
        
        return (
            self.types == other.types and
            np.array_equal(self.sizes, other.sizes) and
            np.array_equal(self.capacity_fractions, other.capacity_fractions)
        )

@dataclass()
class BaseStationConfig:
    """
    The configuration settings of a base station.

    Parameters
    ----------
    bit_loader : type[BitLoader]
        The type of the bit loader (the concrete bit loader class).
    mapper : type[Mapper]
        The type of the mapper (the concrete mapper class).
    precoder : type[Precoder]
        The type of the precoder (the concrete precoder class).
    """
    bit_loader: type["BitLoader"]
    mapper: type["Mapper"]
    precoder: type["Precoder"]

@dataclass()
class UserTerminalConfig:
    """
    The configuration settings of a user terminal.

    Parameters
    ----------
    combiner : type[Combiner]
        The type of the combiner (the concrete combiner class).
    equalizer : type[Equalizer]
        The type of the equalizer (the concrete equalizer class).
    detector : type[Detector]
        The type of the detector (the concrete detector class).
    demapper : type[Demapper]
        The type of the demapper (the concrete demapper class).
    """
    combiner: type["Combiner"]
    equalizer: type["Equalizer"]
    detector: type["Detector"]
    demapper: type["Demapper"]

@dataclass()
class ChannelConfig:
    """
    The configuration settings of a channel.

    Parameters
    ----------
    channel_model : type[ChannelModel]
        The type of the channel model (the concrete channel model class).
    noise_model : type[NoiseModel]
        The type of the noise model (the concrete noise model class).
    """
    channel_model: type["ChannelModel"]
    noise_model: type["NoiseModel"]

@dataclass()
class SystemConfig:
    """
    The configuration settings of a MU-MIMO system.

    Parameters
    ----------
    Pt : float
        The total available transmit power (in Watt).
    B : float
        The system frequency bandwidth (in Hertz).
    
    K : int
        The number of user terminals.
    Nr : int
        The number of receive antennas per user terminal.
    Nt : int
        The number of transmit antennas at the base station.

    c_configs : ConstConfig
        The constellation configuration settings for each UT.
    
    base_station_configs : BaseStationConfig
        The configuration settings of the base station.
    user_terminal_configs : UserTerminalConfig
        The configuration settings of the user terminals.
    channel_configs : ChannelConfig
        The configuration settings of the channel.
    
    name : str
        The name of the system configuration.
    """

    Pt: float
    B: float

    K: int
    Nr: int
    Nt: int

    c_configs: ConstConfig

    base_station_configs: BaseStationConfig
    user_terminal_configs: UserTerminalConfig
    channel_configs: ChannelConfig

    name: str

    def display(self):
        """
        Display system configuration settings in a readable format.

        Returns
        -------
        str_display : str
            A formatted string summarizing the system configuration settings.
        """

        lines: list[str] = []
        
        lines.append(f"  {self.name}:\n")
        lines.append(f"  K  = {self.K} UTs, Nr = {self.Nr}, Nt = {self.Nt}")
        lines.append(f"  Pt = {self.Pt} W, B = {self.B} Hz")
        lines.append(f"  Precoder  : {self.base_station_configs.precoder.__name__}")
        lines.append(f"  Combiner  : {self.user_terminal_configs.combiner.__name__}")
        lines.append(f"  BitLoader : {self.base_station_configs.bit_loader.__name__}")
        lines.append(f"  Channel   : {self.channel_configs.channel_model.__name__}")
        lines.append(f"  Noise     : {self.channel_configs.noise_model.__name__}")
        lines.append("-" * 60)

        str_display = "\n".join(lines)
        return str_display

    def __post_init__(self):

        # Validate dimensions.
        if self.K <= 0: raise ValueError("The number of UTs must be a positive integer.")
        if self.Nt <= 0: raise ValueError("The number of transmit antennas must be a positive integer.")
        if self.Nr <= 0: raise ValueError("The number of receive antennas per UT must be a positive integer.")
        if self.K * self.Nr > self.Nt: raise ValueError("The BS must have at least as many transmit antennas as the total number of receive antennas across all UTs.")

        # Validate constellation types.
        if isinstance(self.c_configs.types, str):
            self.c_configs.types = [self.c_configs.types] * self.K
        elif len(self.c_configs.types) != self.K:
            raise ValueError("The number of constellation types must match K.")

        # Validate constellation sizes.
        if isinstance(self.c_configs.sizes, int):
            self.c_configs.sizes = np.array([self.c_configs.sizes] * self.K, dtype=int)
        elif isinstance(self.c_configs.sizes, np.ndarray) and len(self.c_configs.sizes) != self.K:
            raise ValueError("The number of constellation sizes must match K.")

        # Validate capacity fractions.
        if isinstance(self.c_configs.capacity_fractions, (int, float)):
            self.c_configs.capacity_fractions = np.array([self.c_configs.capacity_fractions] * self.K, dtype=float)
        elif isinstance(self.c_configs.capacity_fractions, np.ndarray):
            if len(self.c_configs.capacity_fractions) != self.K:
                raise ValueError("The number of capacity fractions must match K.")
            if not (np.all(self.c_configs.capacity_fractions >= 0) and np.all(self.c_configs.capacity_fractions <= 1)):
                raise ValueError("Capacity fractions must be between 0 and 1.")
        
    def __eq__(self, other: object) -> bool:
    
        if not isinstance(other, SystemConfig):
            return NotImplemented
        
        return (
            self.Pt == other.Pt and
            self.B == other.B and
            self.K == other.K and
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.c_configs == other.c_configs and
            self.base_station_configs == other.base_station_configs and
            self.user_terminal_configs == other.user_terminal_configs and
            self.channel_configs == other.channel_configs
        )

@dataclass
class SimConfig:
    """
    The configuration settings of a simulation.

    Parameters
    ----------
    snr_dB_values : RealArray
        The SNR values in dB.
    snr_values : RealArray
        The SNR values in linear scale. It is derived from snr_dB_values.
    num_channel_realizations : int
        The minimum number of channel realizations per SNR value.
    num_bit_errors : int
        The minimum number of bit errors per SNR value.
    num_bit_errors_scope : Literal["system-wide", "uts", "streams"]
        The scope over which the minimum number of bit errors are considered.
    M : int
        The number of symbol vector transmissions for each channel realization.
    name : str
        The name of the simulation configuration.
    """

    snr_dB_values: RealArray
    num_channel_realizations: int
    num_bit_errors: int
    num_bit_errors_scope: Literal["system-wide", "uts", "streams"]
    M: int

    name: str

    @property
    def snr_values(self) -> RealArray:
        return 10 ** (self.snr_dB_values / 10)

    def display(self):
        """
        Display simulation configuration settings in a readable format.

        Returns
        -------
        str_display : str
            A formatted string summarizing the simulation configuration settings.
        """

        lines: list[str] = []

        lines.append(f"  {self.name}:\n")
        lines.append(f"  SNR range                  : {self.snr_dB_values[0]} - {self.snr_dB_values[-1]} dB")
        lines.append(f"  Min channel realizations   : {self.num_channel_realizations}")
        lines.append(f"  Min bit errors             : {self.num_bit_errors} ({self.num_bit_errors_scope})")
        lines.append(f"  M                          : {self.M} transmissions per channel")
        lines.append("-" * 60)

        str_display = "\n".join(lines)
        return str_display

    def __post_init__(self):
        if self.num_channel_realizations <= 0: raise ValueError("The minimum number of channel realizations must be a positive integer.")
        if self.num_bit_errors <= 0: raise ValueError("The minimum number of bit errors per SNR value must be a positive integer.")
        if self.M <= 0: raise ValueError("The minimum number of symbol vector transmissions for each channel realization must be a positive integer.")

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, SimConfig):
            return NotImplemented
        
        return (
            np.array_equal(self.snr_dB_values, other.snr_dB_values) and
            self.num_channel_realizations == other.num_channel_realizations and
            self.num_bit_errors == other.num_bit_errors and
            self.num_bit_errors_scope == other.num_bit_errors_scope and
            self.M == other.M
        )


def setup_sim_configs(ref_numbers: list[str], filepath: Path) -> dict[str, SimConfig]:
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

    sim_configs = {}
    for ref_number in ref_numbers:
        config_settings = _load_sim_config(filepath, ref_number)
        sim_config = _create_sim_config(config_settings)
        sim_configs[ref_number] = sim_config

    return sim_configs

def setup_sys_configs(ref_numbers: list[str], filepath: Path) -> dict[str, SystemConfig]:
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

    system_configs = {}
    for ref_number in ref_numbers:
        config_settings = _load_sys_config(filepath, ref_number)
        system_config = _create_sys_config(config_settings)
        system_configs[ref_number] = system_config

    return system_configs



__all__ = [
    "ConstConfig", 
    "BaseStationConfig", "UserTerminalConfig", "ChannelConfig",
    "SystemConfig", "SimConfig",
    "setup_sim_configs", "setup_sys_configs",
]
