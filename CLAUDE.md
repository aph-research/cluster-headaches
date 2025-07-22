# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cluster headache burden simulation project that models the global burden of cluster headache pain through statistical simulations and interactive visualizations. The project consists of:

- **Streamlit web application**: Interactive dashboard for running simulations and visualizing results
- **Scientific simulation engine**: Monte Carlo simulations of cluster headache patterns and pain burden
- **Jupyter notebooks**: Research analysis and paper figure generation
- **Keep-alive infrastructure**: Automated system to maintain Streamlit app availability

## Development Commands

### Running the Application
```bash
streamlit run Cluster_headache_app.py
```

### Python Dependencies
Install dependencies with:
```bash
pip install -r requirements.txt
```

Core dependencies: numpy, scipy, matplotlib, plotly, pandas, streamlit

### Jupyter Notebooks
Open notebooks directly:
- `Cluster headache full model.ipynb` - Complete simulation model
- `Cluster headache paper plots.ipynb` - Publication figure generation  
- `Cluster headache simulations.ipynb` - Simulation experiments

## Architecture

### Core Components

**SimulationConfig** (`SimulationConfig.py`): Configuration dataclass containing all simulation parameters including prevalence rates, treatment percentages, and transformation methods.

**Patient Models** (`models.py`): 
- `Patient` class: Represents individual cluster headache sufferers (chronic/episodic, treated/untreated)
- `Attack` class: Models individual headache attacks with duration and intensity profiles
- Pre-generates attack pools for performance optimization

**Simulation Engine** (`simulation.py`):
- `Simulation` class: Orchestrates Monte Carlo simulations
- Calculates global population groups, generates virtual patients, simulates annual attack patterns
- Computes person-years of pain burden across intensity levels

**Visualization** (`visualizer.py`): Creates interactive Plotly charts for burden analysis, comparative studies, and 3D scatter plots.

**Statistical Utilities** (`stats_utils.py`): Probability distributions for attack frequencies, durations, and intensities based on clinical literature.

### Data Flow

1. **Configuration**: User sets parameters via Streamlit sidebar (prevalence, treatment rates, simulation size)
2. **Population Generation**: Creates patient cohorts based on epidemiological data
3. **Attack Simulation**: Generates realistic attack patterns using statistical distributions
4. **Burden Calculation**: Computes person-years across pain intensity scales
5. **Visualization**: Renders interactive charts comparing different scenarios and transformations

### Keep-Alive System

The project includes automated infrastructure to keep the Streamlit app active:
- **GitHub Action** (`.github/workflows/keep-alive.yml`): Scheduled workflow every 10 hours
- **Probe Scripts**: `keep_awake_script.py` (Selenium) and `probe-action/` (Docker + Puppeteer)
- Prevents Streamlit Community Cloud from hibernating the application

## Key Files

- `Cluster_headache_app.py`: Main Streamlit application entry point
- `models.py`: Patient and Attack data models  
- `simulation.py`: Core simulation logic
- `visualizer.py`: Plotly chart generation
- `stats_utils.py`: Statistical distributions and calculations
- `figs.py`: Figure export utilities