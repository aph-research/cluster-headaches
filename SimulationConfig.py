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
    max_value: int = 100
    power: float = 2.0