"""Helper functions for visualizations."""
from typing import List, Dict


def format_percentage(value: float) -> str:
    """Format a value as a percentage."""
    return f"{value * 100:.1f}%"


def format_score(value: float) -> str:
    """Format a score value."""
    return f"{value:.2f}"


def get_confidence_color(confidence: float) -> str:
    """
    Get color code for confidence level.
    
    Args:
        confidence: Confidence value between 0 and 1
    
    Returns:
        Color name or hex code
    """
    if confidence >= 0.7:
        return "green"
    elif confidence >= 0.4:
        return "orange"
    else:
        return "red"


def get_emoji_for_scenario(scenario_type: str) -> str:
    """Get emoji for scenario type."""
    emojis = {
        "best_case": "🌤️",
        "worst_case": "🌧️",
        "most_likely": "☁️"
    }
    return emojis.get(scenario_type, "📊")


def get_severity_icon(severity: str) -> str:
    """Get icon/emoji for risk severity."""
    icons = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢"
    }
    return icons.get(severity.lower(), "⚠️")
