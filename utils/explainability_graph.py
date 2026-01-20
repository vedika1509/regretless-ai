"""Generate explainability graph showing causal relationships."""
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from services.causal_graph import CausalGraph
from models.decision import StructuredDecision


def create_explainability_graph(
    decision: Optional[StructuredDecision] = None,
    causal_graph: Optional[CausalGraph] = None
) -> go.Figure:
    """
    Create interactive network graph showing causal relationships.
    
    Args:
        decision: StructuredDecision object (optional)
        causal_graph: CausalGraph object (optional)
        
    Returns:
        Plotly figure with network graph
    """
    if causal_graph is None:
        causal_graph = CausalGraph()
    
    if decision is None:
        # Create a demo decision structure
        from models.decision import StructuredDecision, VariableDistribution
        decision = StructuredDecision(
            decision_type="demo",
            variables={},
            raw_text="Demo"
        )
    
    # Build causal graph
    graph = causal_graph.build_graph(decision)
    
    # Extract nodes and edges
    nodes = []
    node_positions = {}
    node_ids = {}
    node_idx = 0
    
    # Add variable nodes (input layer)
    variable_nodes = list(decision.variables.keys()) if decision.variables else []
    for var in variable_nodes:
        nodes.append({
            "id": var,
            "label": var.replace('_', ' ').title(),
            "type": "variable",
            "x": 0,
            "y": node_idx * 2 - len(variable_nodes)
        })
        node_ids[var] = node_idx
        node_positions[var] = (0, node_idx * 2 - len(variable_nodes))
        node_idx += 1
    
    # Add outcome nodes (output layer)
    outcome_nodes = [
        "financial_satisfaction",
        "job_security", 
        "overall_satisfaction",
        "satisfaction_score",
        "risk_score",
        "stress"
    ]
    outcome_start_idx = node_idx
    
    for outcome in outcome_nodes:
        if outcome in graph and graph[outcome]:
            nodes.append({
                "id": outcome,
                "label": outcome.replace('_', ' ').title(),
                "type": "outcome",
                "x": 2,
                "y": node_idx * 2 - len(outcome_nodes)
            })
            node_ids[outcome] = node_idx
            node_positions[outcome] = (2, node_idx * 2 - len(outcome_nodes))
            node_idx += 1
    
    # Create edges
    edges = []
    edge_x = []
    edge_y = []
    
    for outcome_dim, relationships in graph.items():
        if outcome_dim not in node_ids:
            continue
        
        outcome_idx = node_ids[outcome_dim]
        outcome_pos = node_positions[outcome_dim]
        
        for var_name, impact_type, strength, duration in relationships:
            if var_name not in node_ids:
                continue
            
            var_idx = node_ids[var_name]
            var_pos = node_positions[var_name]
            
            # Edge from variable to outcome
            edge_x.extend([var_pos[0], outcome_pos[0], None])
            edge_y.extend([var_pos[1], outcome_pos[1], None])
            
            edges.append({
                "from": var_name,
                "to": outcome_dim,
                "impact": impact_type,
                "strength": strength,
                "duration": duration
            })
    
    # Create node trace
    variable_x = [n["x"] for n in nodes if n["type"] == "variable"]
    variable_y = [n["y"] for n in nodes if n["type"] == "variable"]
    variable_labels = [n["label"] for n in nodes if n["type"] == "variable"]
    
    outcome_x = [n["x"] for n in nodes if n["type"] == "outcome"]
    outcome_y = [n["y"] for n in nodes if n["type"] == "outcome"]
    outcome_labels = [n["label"] for n in nodes if n["type"] == "outcome"]
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2, color='lightgray'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add variable nodes (inputs)
    fig.add_trace(go.Scatter(
        x=variable_x,
        y=variable_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color='#4CAF50',
            line=dict(width=2, color='darkgreen')
        ),
        text=variable_labels,
        textposition="middle right",
        textfont=dict(size=10),
        name="Variables",
        hovertemplate="<b>%{text}</b><br>Variable<extra></extra>"
    ))
    
    # Add outcome nodes (outputs)
    fig.add_trace(go.Scatter(
        x=outcome_x,
        y=outcome_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color='#2196F3',
            line=dict(width=2, color='darkblue')
        ),
        text=outcome_labels,
        textposition="middle left",
        textfont=dict(size=10),
        name="Outcomes",
        hovertemplate="<b>%{text}</b><br>Outcome<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title="Decision Explainability Graph<br><sub>Causal relationships between variables and outcomes</sub>",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=100),
        annotations=[
            dict(
                text="Variables → Outcomes",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,
                xanchor="center",
                yanchor="bottom"
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        template="plotly_white"
    )
    
    return fig
