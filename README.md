# Global Burden of Cluster Headache Pain

A scientific simulation tool for modeling and visualizing the global burden of cluster headache pain through Monte Carlo simulations and interactive data visualization.

## 🌐 Live Demo

**[View the interactive application](https://cluster-headaches.streamlit.app/)**

## Overview

This project provides a comprehensive simulation framework for understanding the global impact of cluster headaches, one of the most painful neurological conditions. Through statistical modeling based on clinical literature, the application generates estimates of:

- Person-years of pain burden across different intensity levels
- Comparative analysis between episodic and chronic cluster headaches
- Treatment impact on pain burden
- Comparison with migraine pain burden
- Global prevalence and burden estimates

## Features

### 🎛️ Interactive Parameters
- Adjustable prevalence rates (26-95 per 100,000 adults)
- Treatment percentage configurations
- Chronic vs episodic case ratios
- Simulation scale (percentage of global population)

### 📊 Advanced Visualizations
- 3D scatter plots of patient pain profiles
- Person-years analysis across intensity scales
- Global burden comparisons
- High-intensity pain burden analysis
- Cluster headache vs migraine comparisons
- Burden ratio heatmaps

### 🔧 Pain Scale Transformations
- Linear scaling
- Piecewise linear transformations
- Power law scaling
- Exponential transformations

## Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/cluster-headaches.git
cd cluster-headaches
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run Cluster_headache_app.py
```

The application will open in your browser at `http://localhost:8501`

## Repository Structure

```
├── Cluster_headache_app.py      # Main Streamlit application
├── models.py                    # Patient and Attack data models
├── simulation.py                # Core simulation engine
├── SimulationConfig.py          # Configuration parameters
├── visualizer.py               # Plotly visualization components
├── stats_utils.py              # Statistical distributions
├── figs.py                     # Figure export utilities
├── keep_awake_script.py        # Selenium keep-alive script
├── requirements.txt            # Python dependencies
├── Cluster headache full model.ipynb       # Complete model notebook
├── Cluster headache paper plots.ipynb     # Publication figures
├── Cluster headache simulations.ipynb     # Simulation experiments
├── probe-action/               # Docker-based keep-alive system
└── .github/workflows/          # GitHub Actions for app maintenance
```

## Scientific Foundation

The simulation is built on epidemiological data and clinical studies:

- **Prevalence**: 26-95 cases per 100,000 adults globally
- **Chronicity**: ~20% of cases become chronic
- **Treatment Access**: ~43% of patients receive appropriate treatment
- **Attack Patterns**: Based on published clinical observations
- **Pain Intensity**: Distributions derived from patient-reported outcomes

## Research Applications

This tool supports research in:
- Health economics and burden of disease studies
- Treatment impact modeling
- Healthcare resource allocation
- Comparative effectiveness research
- Policy development for neurological conditions

## Jupyter Notebooks

The repository includes research notebooks:

- **Full Model**: Complete simulation implementation with detailed analysis
- **Paper Plots**: Publication-ready figure generation
- **Simulations**: Experimental scenarios and parameter sensitivity analysis

## Keep-Alive Infrastructure

The project includes automated systems to maintain the Streamlit Cloud deployment:
- GitHub Actions workflow (every 10 hours)
- Selenium-based web probing
- Docker containerized health checks

## Contributing

Contributions are welcome! Areas for improvement:
- Additional pain scale transformations
- Enhanced statistical models
- New visualization types
- Performance optimizations
- Extended research capabilities

## License

This project is available under the MIT License. See LICENSE file for details.

## Citation

If you use this tool in research, please cite:

```bibtex
@software{cluster_headache_burden,
  title={Global Burden of Cluster Headache Pain Simulation},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-username/cluster-headaches}
}
```

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This simulation tool is for research and educational purposes. Clinical decisions should always be based on professional medical advice and peer-reviewed literature.