from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationConfig:
    world_adult_population: int = 5_728_759_000
    annual_prevalence_per_100k: int = 53
    prop_chronic: float = 0.20
    prop_episodic: float = 1 - prop_chronic
    prop_treated: float = 0.48
    prop_untreated: float = 1 - prop_treated
    percent_of_patients_to_simulate: float = 0.02
    transformation_method: str = 'linear'
    transformation_display: str = 'Linear'
    max_value: int = 1
    power: float = 2.0
    base: float = np.e
    n_taylor: int = 2
    scaling_factor: float = 1.0
    migraine_mean: float = 3.5
    migraine_median: float = 3.7
    migraine_std: float = 1.0
    migraine_prevalence_percentage: float = 0.144
    migraine_fraction_of_year_in_attacks: float = .085