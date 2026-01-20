"""Chart generation utilities."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict


def plot_distribution(results: List[Dict[str, float]], metric: str = "overall_score"):
    """Plot distribution histogram of simulation results."""
    scores = [r[metric] for r in results]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=50,
        marker_color='#FF4B4B',
        opacity=0.7,
        name=metric.replace('_', ' ').title()
    ))
    
    fig.update_layout(
        title=f"Distribution of {metric.replace('_', ' ').title()}",
        xaxis_title="Score",
        yaxis_title="Frequency",
        template="plotly_white",
        height=300
    )
    
    return fig


def plot_scenario_comparison(scenarios: Dict):
    """Plot radar chart comparing scenarios."""
    scenario_names = ["Best Case", "Most Likely", "Worst Case"]
    metrics = ["Financial", "Satisfaction", "Overall", "Low Risk"]
    
    fig = go.Figure()
    
    for scenario_type, scenario_title in zip(
        ["best_case", "most_likely", "worst_case"], 
        scenario_names
    ):
        if scenario_type in scenarios:
            scenario = scenarios[scenario_type]
            values = [
                scenario.outcomes.financial_score,
                scenario.outcomes.satisfaction_score,
                scenario.outcomes.overall_score,
                1 - scenario.outcomes.risk_score  # Low risk is good
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=scenario_title
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Scenario Comparison",
        height=400
    )
    
    return fig


def plot_metric_distributions(results: List[Dict[str, float]]):
    """Plot distributions for all key metrics."""
    metrics = ["overall_score", "financial_score", "satisfaction_score", "risk_score"]
    metric_labels = ["Overall Score", "Financial Score", "Satisfaction Score", "Risk Score"]
    
    fig = go.Figure()
    
    colors = ['#FF4B4B', '#4CAF50', '#2196F3', '#FF9800']
    
    for metric, label, color in zip(metrics, metric_labels, colors):
        scores = [r[metric] for r in results]
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=30,
            name=label,
            opacity=0.7,
            marker_color=color
        ))
    
    fig.update_layout(
        title="Distribution of All Metrics",
        xaxis_title="Score",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        barmode='overlay'
    )
    
    return fig


def plot_timeline_projection(scenarios: Dict):
    """Plot time-based projection (optional)."""
    # This would show how outcomes might evolve over time
    # For now, return a placeholder
    fig = go.Figure()
    fig.add_annotation(
        text="Timeline projections coming soon!",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(height=300)
    return fig
