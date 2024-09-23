from dataclasses import dataclass
import numpy as np
from scipy.stats import lognorm
from stats_utils import (
    generate_bouts_per_year, 
    generate_chronic_active_days, 
    generate_attacks_per_day, 
    generate_attack_duration, 
    generate_max_pain_intensity,
    optimal_sigma, optimal_mu
)

@dataclass
class Attack:
    total_duration: int
    max_intensity: float
    max_intensity_duration: int

class Patient:
    def __init__(self, is_chronic, is_treated):
        self.is_chronic = is_chronic
        self.is_treated = is_treated
        self.attacks = []
        self.generate_profile()
        self.pre_generate_attack_pool()

    def generate_profile(self):
        if self.is_chronic:
            self.active_days = generate_chronic_active_days()
        else:
            self.annual_bouts = generate_bouts_per_year().rvs()
            self.bout_durations = self.generate_bout_durations()

    def pre_generate_attack_pool(self):
        # Estimate the maximum number of attacks in a year
        if self.is_chronic:
            max_attacks = self.active_days * 8  # Assuming max 8 attacks per day
        else:
            max_attacks = sum(self.bout_durations) * 8

        # Generate a pool of attacks
        max_intensities = generate_max_pain_intensity(is_treated=self.is_treated, size=max_attacks)
        total_durations = generate_attack_duration(self.is_chronic, self.is_treated, max_intensities, size=max_attacks)
        # Assuming onset and offset phases take up 15% of the total attack duration each
        max_intensity_durations = np.round(0.7 * total_durations).astype(int)

        self.attack_pool = [Attack(total_durations[i], max_intensities[i], max_intensity_durations[i])
                            for i in range(max_attacks)]
        self.pool_index = 0
        
    def generate_bout_durations(self):
        # Use the lognormal distribution for bout durations
        n_bouts = np.ceil(self.annual_bouts)
        durations = lognorm.rvs(s=optimal_sigma, scale=np.exp(optimal_mu), size=int(n_bouts))
        
        # Adjust the last bout duration if annual_bouts is not an integer
        if self.annual_bouts != int(self.annual_bouts):
            durations[-1] *= (self.annual_bouts - int(self.annual_bouts))
        
        return [max(1, int(duration * 7)) for duration in durations]  # Convert weeks to days, ensure at least 1 day

    def generate_year_of_attacks(self):
        self.attacks = []
        total_attacks = 0

        if self.is_chronic:
            active_days = min(365, self.active_days)
            attacks_per_day = generate_attacks_per_day(self.is_chronic, self.is_treated, size=active_days)
        else:
            active_days = sum(self.bout_durations)
            attacks_per_day = generate_attacks_per_day(self.is_chronic, self.is_treated, size=active_days)

        total_attacks = self.generate_day_attacks(attacks_per_day)
        return total_attacks

    def generate_day_attacks(self, attacks_per_day):
        daily_attacks = 0

        if self.pool_index + sum(attacks_per_day) > len(self.attack_pool):
            # If we've used all pre-generated attacks, generate more
            self.pre_generate_attack_pool()
        
        for attacks_today in attacks_per_day:
            day_attacks = self.attack_pool[self.pool_index:self.pool_index + attacks_today]
            self.attacks.extend(day_attacks)
            self.pool_index += attacks_today
            daily_attacks += attacks_today

        return daily_attacks

    def calculate_intensity_minutes(self):
        intensity_minutes = {}
        for attack in self.attacks:
            intensity = round(attack.max_intensity, 1)  # Round to nearest 0.1
            intensity_minutes[intensity] = intensity_minutes.get(intensity, 0) + attack.max_intensity_duration
        return intensity_minutes

    def calculate_total_attacks(self):
        return len(self.attacks)

    def calculate_total_duration(self):
        return sum(attack.total_duration for attack in self.attacks)

    def calculate_average_intensity(self):
        if not self.attacks:
            return 0
        return np.mean([attack.max_intensity for attack in self.attacks])