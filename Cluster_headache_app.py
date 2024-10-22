import streamlit as st
import numpy as np
from SimulationConfig import SimulationConfig
from simulation import Simulation
from visualizer import Visualizer

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)

# Sidebar inputs for simulation parameters
def create_sidebar_inputs():
    st.sidebar.header("Parameters")

    annual_prevalence_per_100k = st.sidebar.slider("Annual prevalence (adults per 100,000)", 26, 95, 53, 1)

    total_ch_sufferers = SimulationConfig.world_adult_population * (annual_prevalence_per_100k / 100000)
    
    st.sidebar.write(f"Total individuals with cluster headaches annually worldwide: {int(total_ch_sufferers):,}")
    
    prop_chronic = st.sidebar.slider("Percentage of chronic cases", 0, 100, 20, format="%d%%") / 100
    prop_treated = st.sidebar.slider("Percentage of treated cases", 0, 100, 48, format="%d%%") / 100

    percent_of_patients_to_simulate = st.sidebar.slider("Percentage of worldwide individuals to simulate", 
                                                        0.01, 0.1, 0.02, 
                                                        format="%.2f%%")

    theme = st.get_option('theme.base')

    return SimulationConfig(
        annual_prevalence_per_100k=annual_prevalence_per_100k,
        prop_chronic=prop_chronic,
        prop_episodic=1 - prop_chronic,
        prop_treated=prop_treated,
        prop_untreated=1 - prop_treated,
        percent_of_patients_to_simulate=percent_of_patients_to_simulate,
        theme=theme
    )

# Sidebar inputs for intensity scale transformation parameters
def create_intensity_scale_inputs(config):
    with st.sidebar.expander("Intensity Scale Transformation"):
        method_map = {
            'Linear': 'linear',
            'Piecewise Linear': 'piecewise_linear',
            'Power': 'power',
            'Exponential': 'exponential',
        }
        transformation_display = st.selectbox(
            "Transformation method",
            list(method_map.keys())
        )
        transformation_method = method_map[transformation_display]
        
        if transformation_method == 'power':
            power = st.number_input("Power", min_value=1.0, max_value=100.0, value=SimulationConfig.power, step=0.1)
        else:
            power = SimulationConfig.power

        if transformation_method == 'exponential':
            base = st.number_input("Base", min_value=1.0, max_value=20.0, value=SimulationConfig.base, step=1.0)
            scaling_factor = st.number_input("Scaling factor", min_value=0.01, max_value=100.0, value=SimulationConfig.scaling_factor, step=0.01)
        else:
            base = SimulationConfig.base
            scaling_factor = SimulationConfig.scaling_factor

    config.transformation_method = transformation_method
    config.transformation_display = transformation_display
    config.power = power
    config.base = base
    config.scaling_factor = scaling_factor

    return config

def create_migraine_inputs(config):
    with st.sidebar.expander("Migraine Parameters"):
        migraine_mean = st.number_input("Mean pain intensity", min_value=1.0, max_value=9.0, value=config.migraine_mean, step=0.1)
        migraine_median = st.number_input("Median pain intensity", min_value=1.0, max_value=9.0, value=config.migraine_median, step=0.1)
        migraine_std = st.number_input("Standard deviation", min_value=0.1, max_value=4.0, value=config.migraine_std, step=0.1)

    config.migraine_mean = migraine_mean
    config.migraine_median = migraine_median
    config.migraine_std = migraine_std

    return config

# Main function to run the app
def main():
    st.title("Global Burden of Cluster Headache Pain")

    # set_random_seeds()

    # Sidebar: configure simulation parameters
    config = create_sidebar_inputs()
    
    if 'simulation' not in st.session_state:
        simulation = Simulation(config)
    else:
        simulation = st.session_state.simulation
        simulation.config = config

    simulation.calculate_ch_groups()

    # Display simulated patients info
    total_simulated, group_info = simulation.get_simulated_patients_info()
    st.sidebar.write(f"Total individuals to simulate: {total_simulated:,}, of which:")
    for group, count, percentage in group_info:
        st.sidebar.write(f"- {group}: {count:,} ({percentage}%)")

    run_simulation = st.sidebar.button("Run Simulation")

    # Intensity Scale Transformation inputs
    config = create_intensity_scale_inputs(config)
    config = create_migraine_inputs(config)

    if run_simulation:
        with st.spinner("Running simulation..."):
            simulation = Simulation(config)
            simulation.run()
            st.session_state.simulation = simulation
            st.session_state.simulation_run = True

    # If simulation has been run, process and display results
    if 'simulation_run' in st.session_state and st.session_state.simulation_run:
        visualizer = Visualizer(simulation)
    
        # Visualization sections
        fig_avg = visualizer.create_average_minutes_plot()
        st.plotly_chart(fig_avg)

        fig_global = visualizer.create_global_person_years_plot()
        st.plotly_chart(fig_global)
        
        fig_3d_patients = visualizer.create_3d_patient_scatter()
        st.plotly_chart(fig_3d_patients, use_container_width=True)

        fig_total = visualizer.create_total_person_years_plot()
        st.plotly_chart(fig_total)

        fig_high_intensity = visualizer.create_high_intensity_person_years_plot()
        st.plotly_chart(fig_high_intensity)

        fig_comparison = visualizer.create_comparison_plot()
        st.plotly_chart(fig_comparison)

        simulation.update_transformation_params(config.transformation_method, 
                                                config.transformation_display,
                                                config.power,
                                                config.base,
                                                config.scaling_factor,
                                                config.migraine_mean, 
                                                config.migraine_median,
                                                config.migraine_std)
        
        fig_adjusted = visualizer.create_adjusted_pain_units_plot()
        st.plotly_chart(fig_adjusted)
        
        # Update the table dynamically based on transformation parameters
        df = visualizer.create_summary_table()
        visualizer.display_summary_table(df)

        fig_migraine = visualizer.plot_ch_vs_migraine_person_years()
        st.plotly_chart(fig_migraine)

        fig_migraine_comparison = visualizer.create_adjusted_pain_units_plot_comparison_migraine()
        st.plotly_chart(fig_migraine_comparison, use_container_width=True)
        
        fig_migraine_comparison_3d, fig_intensities = visualizer.create_adjusted_pain_units_plot_comparison_migraine_3d()
        st.plotly_chart(fig_migraine_comparison_3d, use_container_width=True)
        if fig_intensities.data:
            st.plotly_chart(fig_intensities)

        pain_threshold = 7.0
        fig_migraine_comparison_threshold = visualizer.create_adjusted_pain_units_plot_comparison_migraine(pain_threshold)
        st.plotly_chart(fig_migraine_comparison_threshold, use_container_width=True)

        fig_migraine_comparison_3d_threshold, fig_intensities_threshold = visualizer.create_adjusted_pain_units_plot_comparison_migraine_3d(pain_threshold)
        st.plotly_chart(fig_migraine_comparison_3d_threshold, use_container_width=True)
        if fig_intensities_threshold.data:
            st.plotly_chart(fig_intensities_threshold)
    else:
        st.info('Please select your parameters (or leave the default ones) and then press "Run Simulation".')

# Run the app
if __name__ == "__main__":
    main()