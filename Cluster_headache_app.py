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

    total_ch_sufferers = SimulationConfig.world_population * SimulationConfig.adult_fraction * (annual_prevalence_per_100k / 100000)
    
    st.sidebar.write(f"Total individuals with cluster headaches annually worldwide: {int(total_ch_sufferers):,}")
    
    prop_chronic = st.sidebar.slider("Percentage of chronic cases", 0, 100, 20, format="%d%%") / 100
    prop_treated = st.sidebar.slider("Percentage of treated cases", 0, 100, 48, format="%d%%") / 100

    percent_of_patients_to_simulate = st.sidebar.slider("Percentage of worldwide individuals to simulate", 
                                                        0.01, 0.1, 0.02, 
                                                        format="%.2f%%")

    return SimulationConfig(
        annual_prevalence_per_100k=annual_prevalence_per_100k,
        prop_chronic=prop_chronic,
        prop_treated=prop_treated,
        percent_of_patients_to_simulate=percent_of_patients_to_simulate,
    )

# Sidebar inputs for intensity scale transformation parameters
def create_intensity_scale_inputs(config):
    with st.sidebar.expander("Intensity Scale Transformation"):
        method_map = {
            'Linear': 'linear',
            'Piecewise Linear': 'piecewise_linear',
            'Power': 'power',
            'Power (scaled)': 'power_scaled',
            'Fitted Exponential': 'custom_exp',
            'Logarithmic': 'log'
        }
        transformation_display = st.selectbox(
            "Select transformation method:",
            list(method_map.keys())
        )
        transformation_method = method_map[transformation_display]
        
        max_value = st.number_input("Select maximum value of the scale:", min_value=10, max_value=500, value=100, step=10)
        
        power = st.slider("Select power:", min_value=1.0, max_value=5.0, value=2.0, step=0.1) if transformation_method in ['power', 'power_scaled'] else 2

    config.transformation_method = transformation_method
    config.transformation_display = transformation_display
    config.max_value = max_value
    config.power = power

    return config

# Main function to run the app
def main():
    st.title("Global Burden of Cluster Headache Pain")

    set_random_seeds()

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
    new_config = create_intensity_scale_inputs(config)
    
    if 'simulation' in st.session_state:
        if (new_config.transformation_method != config.transformation_method or
            new_config.power != config.power or
            new_config.max_value != config.max_value):
            st.session_state.simulation.update_transformation_params(
                new_config.transformation_method,
                new_config.power,
                new_config.max_value
            )
            # Force a rerun to update the display
            st.experimental_rerun()
    
    config = new_config  # Update the config with new values

    if run_simulation:
        with st.spinner("Running simulation..."):
            simulation.run()
            st.session_state.simulation = simulation
            st.session_state.simulation_run = True

    # If simulation has been run, process and display results
    if 'simulation_run' in st.session_state and st.session_state.simulation_run:
        results = st.session_state.simulation.get_results()
        visualizer = Visualizer(results)
    
        # Visualization sections
        fig_avg = visualizer.create_average_minutes_plot()
        st.plotly_chart(fig_avg)

        fig_global = visualizer.create_global_person_years_plot()
        st.plotly_chart(fig_global)
        
        fig_3d_patients = visualizer.create_3d_patient_scatter()
        st.plotly_chart(fig_3d_patients)

        fig_total = visualizer.create_total_person_years_plot()
        st.plotly_chart(fig_total)

        fig_high_intensity = visualizer.create_high_intensity_person_years_plot()
        st.plotly_chart(fig_high_intensity)

        fig_comparison = visualizer.create_comparison_plot()
        st.plotly_chart(fig_comparison)

        fig_adjusted = visualizer.create_adjusted_pain_units_plot(
            transformation_method=config.transformation_method,
            transformation_display=config.transformation_display,
            power=config.power,
            max_value=config.max_value
        )
        st.plotly_chart(fig_adjusted)

        st.subheader("Intensity-Adjusted Pain Units Experienced Annually")
        st.write("(Values in brackets represent adjusted pain units.)")
        
        # Update the table dynamically based on transformation parameters
        df = visualizer.create_summary_table(
            transformation_method=config.transformation_method,
            power=config.power,
            max_value=config.max_value
        )
        visualizer.display_summary_table(df)

    else:
        st.info('Please select your parameters (or leave the default ones) and then press "Run Simulation".')

# Run the app
if __name__ == "__main__":
    main()