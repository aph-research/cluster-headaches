import numpy as np
from scipy.stats import lognorm, gmean, rv_discrete, beta, truncnorm, expon, skewnorm
from scipy.optimize import minimize, curve_fit
from dataclasses import dataclass

def generate_bouts_per_year():
    bout_frequency_datapoints = {
        'Gaul': {'n': 209, 'dist': {1: 0.6, 2: 0.3, 3: 0.1}},
        'Li': {'n': 327, 'dist': {0.5: 0.416, 1: 0.370, 2.5: 0.214}},
        'Friedman': {'n': 50, 'dist': {0.5: 0.46, 1: 0.54}},
        'Ekbom': {'n': 105, 'dist': {0.5: 0.14, 1: 0.40, 2: 0.31, 3: 0.15}},
        'Manzoni': {'n': 161, 'dist': {1: 0.27, 1.5: 0.73}},
        'Sutherland': {'n': 49, 'dist': {0.5: 0.512+0.174, 1: 0.140, 2: 0.174}},
        'Kudrow': {'n': 428, 'dist': {0.5: 0.19, 1: 0.67, 2.5: 0.14}}
    }

    combined_dist = {}
    total_n = sum(datapoint['n'] for datapoint in bout_frequency_datapoints.values())

    for datapoint in bout_frequency_datapoints.values():
        weight = datapoint['n'] / total_n
        for bouts, prob in datapoint['dist'].items():
            combined_dist[bouts] = combined_dist.get(bouts, 0) + prob * weight

    total_prob = sum(combined_dist.values())
    combined_dist = {k: v/total_prob for k, v in combined_dist.items()}

    return rv_discrete(values=(list(combined_dist.keys()), list(combined_dist.values())))

def generate_bout_duration_distribution():
    bout_duration_datapoints = []
    sample_sizes = []

    # Gaul et al. (2012)
    bout_duration_datapoints.append(8.5)
    sample_sizes.append(209)

    # Li et al. (2022)
    total_li = 327
    original_proportions = np.array([0.104, 0.235, 0.502, 0.131])
    sum_proportions = np.sum(original_proportions)
    new_proportions = original_proportions / sum_proportions
    bout_duration_datapoints.extend([1, gmean([2, 4]), gmean([4, 8]), 8])
    sample_sizes.extend([int(prop * total_li) for prop in new_proportions])

    # Friedman & Mikropoulos (1958)
    bout_duration_datapoints.append(gmean([6, 8]))
    sample_sizes.append(50)

    # Ekbom (1970)
    bout_duration_datapoints.append(gmean([4, 12]))
    sample_sizes.append(105)

    # Lance & Anthony (1971)
    bout_duration_datapoints.append(gmean([2, 12]))
    sample_sizes.append(60)

    # Sutherland & Eadie (1972)
    total_sutherland = 58
    bout_duration_datapoints.extend([np.mean([0, 4]), gmean([5, 13]), gmean([14, 26]), gmean([27, 52])])
    sample_sizes.extend([int(0.23 * total_sutherland), int(0.45 * total_sutherland), 
                         int(0.19 * total_sutherland), int(0.14 * total_sutherland)])

    # Rozen & Fishman (2012)
    bout_duration_datapoints.append(10.3)
    sample_sizes.append(101)

    # Manzoni et al. (1983)
    bout_duration_datapoints.append(gmean([4, 8]))
    sample_sizes.append(161)

    # Convert to numpy arrays
    bout_duration_datapoints = np.array(bout_duration_datapoints)
    sample_sizes = np.array(sample_sizes)

    # Use sample sizes as weights
    weights = sample_sizes / np.sum(sample_sizes)

    def neg_log_likelihood(params):
        mu, sigma = params
        return -np.sum(weights * lognorm.logpdf(bout_duration_datapoints, s=sigma, scale=np.exp(mu)))

    initial_params = [np.log(np.average(bout_duration_datapoints, weights=weights)), 0.5]
    result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead')
    return result.x

optimal_mu, optimal_sigma = generate_bout_duration_distribution()

def fit_lognormal(mean, std):
    variance = std**2
    mu = np.log(mean**2 / np.sqrt(variance + mean**2))
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    return mu, sigma

def truncated_lognorm_pdf(x, mu, sigma, upper_bound=np.inf):
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    cdf_upper = lognorm.cdf(upper_bound, s=sigma, scale=np.exp(mu))
    return np.where(x <= upper_bound, pdf / cdf_upper, 0)

def estimate_untreated(treated_mean, treated_std, treatment_effect=1.05):
    cv = treated_std / treated_mean  # Coefficient of variation
    untreated_mean = treated_mean * treatment_effect
    untreated_std = untreated_mean * cv
    return untreated_mean, untreated_std

@dataclass
class AttackParameters:
    episodic_treated_mu: float
    episodic_treated_sigma: float
    episodic_untreated_mu: float
    episodic_untreated_sigma: float
    chronic_treated_mu: float
    chronic_treated_sigma: float
    chronic_untreated_mu: float
    chronic_untreated_sigma: float

def initialize_attack_parameters():
    # Gaul et al. (2012) data for treated patients
    episodic_treated_mean, episodic_treated_std = 3.1, 2.1
    chronic_treated_mean, chronic_treated_std = 3.3, 3.0

    # Estimating untreated values
    episodic_untreated_mean, episodic_untreated_std = estimate_untreated(episodic_treated_mean, episodic_treated_std)
    chronic_untreated_mean, chronic_untreated_std = estimate_untreated(chronic_treated_mean, chronic_treated_std)

    # Fit lognormal distributions
    episodic_treated_mu, episodic_treated_sigma = fit_lognormal(episodic_treated_mean, episodic_treated_std)
    chronic_treated_mu, chronic_treated_sigma = fit_lognormal(chronic_treated_mean, chronic_treated_std)
    episodic_untreated_mu, episodic_untreated_sigma = fit_lognormal(episodic_untreated_mean, episodic_untreated_std)
    chronic_untreated_mu, chronic_untreated_sigma = fit_lognormal(chronic_untreated_mean, chronic_untreated_std)

    return AttackParameters(
        episodic_treated_mu, episodic_treated_sigma,
        episodic_untreated_mu, episodic_untreated_sigma,
        chronic_treated_mu, chronic_treated_sigma,
        chronic_untreated_mu, chronic_untreated_sigma
    )

# Initialize the parameters
attack_params = initialize_attack_parameters()

def generate_attacks_per_day(is_chronic, is_treated, max_daily_ch=np.inf, size=1):
    if is_chronic:
        if is_treated:
            mu, sigma = attack_params.chronic_treated_mu, attack_params.chronic_treated_sigma
        else:
            mu, sigma = attack_params.chronic_untreated_mu, attack_params.chronic_untreated_sigma
    else:
        if is_treated:
            mu, sigma = attack_params.episodic_treated_mu, attack_params.episodic_treated_sigma
        else:
            mu, sigma = attack_params.episodic_untreated_mu, attack_params.episodic_untreated_sigma
    
    attacks = lognorm.rvs(s=sigma, scale=np.exp(mu), size=size)
    
    # Filter out values that exceed max_daily_ch and resample them
    while np.any(attacks > max_daily_ch):
        invalid_indices = attacks > max_daily_ch
        attacks[invalid_indices] = lognorm.rvs(s=sigma, scale=np.exp(mu), size=np.sum(invalid_indices))
    
    return np.round(attacks).astype(int)

def generate_chronic_active_days():
    while True:
        active_days = int(lognorm.rvs(s=.5, scale=np.exp(np.log(200))))
        active_days = min(active_days, 365)
        return active_days

def generate_attack_duration(is_chronic, is_treated, max_intensities, size):
    mu = 4.0 + (0.3 if is_chronic else 0)
    sigma = 0.4
    
    base_durations = lognorm.rvs(s=sigma, scale=np.exp(mu), size=size)
    intensity_factor = 0.1064 * max_intensities + 0.5797
    adjusted_durations = base_durations * intensity_factor

    if is_treated:
        max_effect = 0.3
        intensity_normalized = (max_intensities - 1) / 9
        mean_effect = 1 - (max_effect * intensity_normalized)
        a, b = 5, 2
        treatment_effect = beta.rvs(a, b, size=size) * mean_effect
        adjusted_durations *= treatment_effect
    
    return np.clip(np.round(adjusted_durations).astype(int), 15, 360)

def generate_max_pain_intensity(is_treated, size):
    mean_mild_moderate = 4.0
    sd_mild_moderate = 2.0
    mean_moderate_severe = 7.5
    sd_moderate_severe = 2.0
    scale_very_severe = .7 if is_treated else 0.5
    
    mild_to_moderate = truncnorm.rvs((1-mean_mild_moderate)/sd_mild_moderate, np.inf, loc=mean_mild_moderate, scale=sd_mild_moderate, size=size)
    moderate_to_severe = truncnorm.rvs((1-mean_moderate_severe)/sd_moderate_severe, np.inf, loc=mean_moderate_severe, scale=sd_moderate_severe, size=size)
    very_severe = 10 - expon.rvs(scale=scale_very_severe, size=size)
    
    if is_treated:
        choices = np.random.choice(3, size=size, p=[0.40, 0.35, 0.25])
    else:
        choices = np.random.choice(3, size=size, p=[0.20, 0.50, 0.30])

    intensities = np.where(choices == 0, mild_to_moderate,
                  np.where(choices == 1, moderate_to_severe, very_severe))
    
    return np.round(np.clip(intensities, 1, 10), decimals=1)

def transform_intensity(intensities, method='linear', power=2, max_value=100):
    if method == 'linear':
        return intensities * (max_value / 10)
    elif method == 'power':
        return (intensities ** power) * (max_value / (10 ** power))
    elif method == 'power_scaled':
        return (intensities / 10)**power * max_value
    elif method == 'custom_exp':
        exp_func = lambda x, A, B, C: A * np.exp(B * x) + C
        x = np.array([0, 7.42, 10])
        y = np.array([0, max_value/2, max_value])
        p0 = np.array([max_value/10, 0.5, 0])
        A, B, C = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)[0]
        return exp_func(intensities, A, B, C)
    elif method == 'piecewise_linear':
        breakpoint = 8
        lower_slope = (max_value / 2) / breakpoint
        upper_slope = (max_value / 2) / (10 - breakpoint)
        return np.where(intensities <= breakpoint,
                        lower_slope * intensities,
                        (max_value / 2) + upper_slope * (intensities - breakpoint))
    elif method == 'log':
        return (10**(intensities/2.5) - 1) * (max_value / (10**(10/2.5) - 1))
    else:
        raise ValueError("Invalid method.")

def calculate_adjusted_pain_units(time_amounts, intensities, transformation_method, power, max_value):
    transformed_intensities = transform_intensity(intensities, method=transformation_method, power=power, max_value=max_value)
    return [y * t for y, t in zip(time_amounts, transformed_intensities)]

def calculate_migraine_distribution(migraine_mean, migraine_median, migraine_std):
    # Estimate skewness parameter
    a = -4 * (migraine_mean - migraine_median) / migraine_std
    
    # Define the truncation points
    lower_bound, upper_bound = 0, 10

    # Calculate the CDF at the truncation points
    cdf_lower = skewnorm.cdf(lower_bound, a, loc=migraine_mean, scale=migraine_std)
    cdf_upper = skewnorm.cdf(upper_bound, a, loc=migraine_mean, scale=migraine_std)

    # Compute the normalization factor
    normalization_factor = cdf_upper - cdf_lower

    # Define the bin edges
    bin_edges = np.linspace(0, 10, 101)  # 101 bins between 0 and 10

    # Calculate the PDF values
    pdf_values = skewnorm.pdf(bin_edges, a, loc=migraine_mean, scale=migraine_std)

    # Normalize the PDF values
    normalized_pdf_values = pdf_values / normalization_factor

    return bin_edges, normalized_pdf_values