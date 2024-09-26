from dataclasses import dataclass

@dataclass
class SimulationConfig:
    world_population: int = 8_200_000_000
    adult_fraction: float = 0.72
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
    base: float = 10
    scaling_factor: float = 2.5
    migraine_mean: float = 5.4
    migraine_median: float = 5.8
    migraine_std: float = 1.0
    migraine_prevalence_percentage: float = 0.144
    migraine_fraction_of_year_in_attacks: float = .085