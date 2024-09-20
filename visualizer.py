import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from simulation import calculate_adjusted_pain_units

class Visualizer:
    def __init__(self, simulation_results):
        self.results = simulation_results
        self.intensities = simulation_results['intensities']
        self.group_data = simulation_results['group_data']
        self.global_person_years = simulation_results['global_person_years']
        self.global_std_person_years = simulation_results['global_std_person_years']
        self.ch_groups = simulation_results['ch_groups']

    def create_plot(self, data, title, y_title):
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        markers = ['circle', 'square', 'diamond', 'cross']

        for i, (name, values, std) in enumerate(data):
            color = colors[i % len(colors)]
            rgb_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            marker = markers[i % len(markers)]
            
            # Lower bound of shaded area
            fig.add_trace(go.Scatter(
                x=self.intensities,
                y=[max(0, v - s) for v, s in zip(values, std)],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Upper bound of shaded area
            fig.add_trace(go.Scatter(
                x=self.intensities,
                y=[v + s for v, s in zip(values, std)],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba({rgb_color[0]},{rgb_color[1]},{rgb_color[2]},0.2)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Main line with markers
            fig.add_trace(go.Scatter(
                x=self.intensities,
                y=values,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(
                    symbol=marker,
                    size=[8 if x.is_integer() else 0 for x in self.intensities],
                    color=color,
                ),
                hoverinfo='x+y+name'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Pain Intensity',
            yaxis_title=y_title,
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            yaxis=dict(tickformat=',.0f'),
            legend_title_text='',
            hovermode='closest',
            template='plotly_dark',
            legend=dict(
                itemsizing='constant',
                itemwidth=30,
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1
            )
        )

        return fig

    def create_average_minutes_plot(self):
        avg_data = [(name, avg, std) for name, avg, std, _, _ in self.group_data]
        return self.create_plot(avg_data, 
                                'Average Minutes per Year Spent at Different Pain Intensities (±1σ)',
                                'Average Minutes per Year')

    def create_global_person_years_plot(self):
        global_data = [(name, self.global_person_years[name], self.global_std_person_years[name]) 
                       for name in self.ch_groups.keys()]
        return self.create_plot(global_data,
                                'Global Annual Person-Years Spent in Cluster Headaches by Intensity (±1σ)',
                                'Global Person-Years per Year')

    def create_bar_plot(self, groups, values, errors, title, y_title):
        color_map = {
            'Episodic Treated': px.colors.qualitative.Plotly[0],
            'Episodic Untreated': px.colors.qualitative.Plotly[1],
            'Chronic Treated': px.colors.qualitative.Plotly[2],
            'Chronic Untreated': px.colors.qualitative.Plotly[3]
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=groups,
                y=values,
                error_y=dict(type='data', array=errors, visible=True),
                marker=dict(
                    color=[color_map[group] for group in groups],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            )
        ])
        
        fig.update_layout(
            title=title,
            yaxis_title=y_title,
            template='plotly_dark',
            yaxis=dict(tickformat=',.0f'),
            showlegend=False,
            bargap=0.3
        )
        return fig

    def create_total_person_years_plot(self):
        groups = list(self.ch_groups.keys())
        total_values = [sum(self.global_person_years[name]) for name in groups]
        total_error = [np.sqrt(sum(std**2)) for std in self.global_std_person_years.values()]
        
        return self.create_bar_plot(groups,
                                    total_values,
                                    total_error,
                                    'Total Estimated Person-Years Spent in Cluster Headaches Annually by Group',
                                    'Total Person-Years')

    def create_high_intensity_person_years_plot(self):
        groups = list(self.ch_groups.keys())
        high_intensity_values = [sum(years[90:]) for years in self.global_person_years.values()]
        high_intensity_error = [np.sqrt(sum(std[90:]**2)) for std in self.global_std_person_years.values()]
        
        return self.create_bar_plot(groups,
                                    high_intensity_values,
                                    high_intensity_error,
                                    'Estimated Person-Years Spent in Cluster Headaches Annually by Group (Intensity ≥9/10)',
                                    'Person-Years (Intensity ≥9/10)')

    def create_comparison_plot(self):
        total_all_groups = sum(sum(years) for years in self.global_person_years.values())
        total_all_groups_std = np.sqrt(sum(sum(std**2) for std in self.global_std_person_years.values()))
        
        high_intensity_all_groups = sum(sum(years[90:]) for years in self.global_person_years.values())
        high_intensity_all_groups_std = np.sqrt(sum(sum(std[90:]**2) for std in self.global_std_person_years.values()))
        
        bar_values = [total_all_groups, high_intensity_all_groups]
        bar_errors = [total_all_groups_std, high_intensity_all_groups_std]
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Total Person-Years', 'Person-Years at ≥9/10 Intensity'],
                y=bar_values,
                error_y=dict(type='data', array=bar_errors, visible=True),
                marker=dict(
                    color=['blue', 'red'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            )
        ])

        fig.update_layout(
            title='Comparison of Total and ≥9/10 Intensity Person-Years Across All Groups',
            yaxis_title='Person-Years',
            template='plotly_dark',
            yaxis=dict(tickformat=',.0f'),
            showlegend=False,
            bargap=0.3
        )

        return fig
    
    def create_adjusted_pain_units_plot(self, transformation_method, transformation_display, power, max_value):
        adjusted_data = []
        for name in self.results['ch_groups'].keys():
            values = calculate_adjusted_pain_units(
                self.results['global_person_years'][name],
                self.results['intensities'],
                transformation_method,
                power,
                max_value
            )
            std = [0] * len(values)
            adjusted_data.append((name, values, std))

        fig_adjusted = go.Figure()
        fig_adjusted = self.create_plot(
            adjusted_data,
            title=f"Intensity-Adjusted Pain Units by Cluster Headache Group ({transformation_display} Transformation)",
            y_title="Intensity-Adjusted Pain Units"
        )

        max_adjusted_value = max(max(values) for _, values, _ in adjusted_data)
        fig_adjusted.update_layout(yaxis=dict(range=[0, max_adjusted_value * 1.1]))

        return fig_adjusted
        
    def create_summary_table(self, transformation_method, power, max_value):
        def format_with_adjusted(value, adjusted):
            return f"{value:,.0f} ({adjusted:,.0f})"
    
        table_data = []
        total_row = {
            'Group': 'Total',
            'Average Patient': {key: 0 for key in ['Minutes', 'High-Intensity Minutes', 'Adjusted Units', 'High-Intensity Adjusted Units']},
            'Global Estimate': {key: 0 for key in ['Person-Years', 'High-Intensity Person-Years', 'Adjusted Units', 'High-Intensity Adjusted Units']}
        }
    
        for group in self.results['ch_groups'].keys():
            avg_data = next(avg for name, avg, _, _, _ in self.results['group_data'] if name == group)
            avg_minutes = sum(avg_data)
            avg_high_minutes = sum(avg_data[90:])
            global_years = sum(self.results['global_person_years'][group])
            global_high_years = sum(self.results['global_person_years'][group][90:])
            
            # Use the calculate_adjusted_pain_units function from the simulation results
            avg_adjusted_units = sum(self.results['calculate_adjusted_pain_units'](avg_data, self.results['intensities'], transformation_method, power, max_value))
            avg_high_adjusted_units = sum(self.results['calculate_adjusted_pain_units'](avg_data[90:], self.results['intensities'][90:], transformation_method, power, max_value))
            
            global_adjusted_units = sum(self.results['calculate_adjusted_pain_units'](self.results['global_person_years'][group], self.results['intensities'], transformation_method, power, max_value))
            global_high_adjusted_units = sum(self.results['calculate_adjusted_pain_units'](self.results['global_person_years'][group][90:], self.results['intensities'][90:], transformation_method, power, max_value))
            
            row = {
                'Group': group,
                'Average Patient': {
                    'Minutes': avg_minutes,
                    'High-Intensity Minutes': avg_high_minutes,
                    'Adjusted Units': avg_adjusted_units,
                    'High-Intensity Adjusted Units': avg_high_adjusted_units
                },
                'Global Estimate': {
                    'Person-Years': global_years,
                    'High-Intensity Person-Years': global_high_years,
                    'Adjusted Units': global_adjusted_units,
                    'High-Intensity Adjusted Units': global_high_adjusted_units
                }
            }
            table_data.append(row)
            
            # Update total row
            for category in ['Average Patient', 'Global Estimate']:
                for key in total_row[category].keys():
                    total_row[category][key] += row[category][key]
        
        table_data.append(total_row)
    
        df_data = [
            {
                'Group': row['Group'],
                'Minutes': format_with_adjusted(row['Average Patient']['Minutes'], row['Average Patient']['Adjusted Units']),
                'High-Intensity Minutes': format_with_adjusted(row['Average Patient']['High-Intensity Minutes'], row['Average Patient']['High-Intensity Adjusted Units']),
                'Person-Years': format_with_adjusted(row['Global Estimate']['Person-Years'], row['Global Estimate']['Adjusted Units']),
                'High-Intensity Person-Years': format_with_adjusted(row['Global Estimate']['High-Intensity Person-Years'], row['Global Estimate']['High-Intensity Adjusted Units'])
            }
            for row in table_data
        ]
        df = pd.DataFrame(df_data)
        
        df.columns = pd.MultiIndex.from_tuples([
            ('', 'Group'),
            ('Average Patient', 'Total Minutes in Pain'),
            ('Average Patient', 'Minutes in ≥9/10 Pain'),
            ('Global Estimate', 'Total Person-Years in Pain'),
            ('Global Estimate', 'Person-Years in ≥9/10 Pain')
        ])
        
        return df

    def display_summary_table(self, df):
        css = """
        <style>
            .dataframe {
                width: 100%;
                text-align: right;
                border-collapse: collapse;
                font-size: 0.9em;
                color: #333;
                background-color: #f8f8f8;
                border-left: none;
                border-right: none;
            }
            .dataframe th, .dataframe td {
                border-top: 1px solid #ddd;
                border-bottom: 1px solid #ddd;
                border-left: none;
                border-right: none;
                padding: 8px;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-width: 150px;
                text-align: center;
            }
            .dataframe thead tr:nth-child(1) th {
                background-color: #e0e0e0;
                text-align: center;
                font-weight: bold;
                color: #333;
            }
            .dataframe thead tr:nth-child(2) th {
                background-color: #e8e8e8;
                text-align: center;
                color: #333;
            }
            .dataframe tbody tr:nth-child(even) {
                background-color: #f0f0f0;
            }
            .dataframe tbody tr:nth-child(odd) {
                background-color: #f8f8f8;
            }
            .dataframe tbody tr:hover {
                background-color: #e8e8e8;
            }
            .dataframe td:first-child, .dataframe th:first-child {
                text-align: left;
            }
            .table-note {
                margin-top: 10px;
                font-style: italic;
                font-size: 0.9em;
            }
            .dataframe tr:last-child {
                font-weight: bold;
            }
            @media (prefers-color-scheme: dark) {
                .dataframe, .table-note {
                    color: #e0e0e0;
                    background-color: #2c2c2c;
                }
                .dataframe th, .dataframe td {
                    border-color: #4a4a4a;
                }
                .dataframe thead tr:nth-child(1) th,
                .dataframe thead tr:nth-child(2) th {
                    background-color: #3c3c3c;
                    color: #e0e0e0;
                }
                .dataframe tbody tr:nth-child(even) {
                    background-color: #323232;
                }
                .dataframe tbody tr:nth-child(odd) {
                    background-color: #2c2c2c;
                }
                .dataframe tbody tr:hover {
                    background-color: #3a3a3a;
                }
            }
        </style>
        """
        
        table_html = df.to_html(index=False, escape=False, classes='dataframe')
        
        st.markdown(css, unsafe_allow_html=True)
        st.write(table_html, unsafe_allow_html=True)

    def update_results(self, new_results):
        self.results = new_results