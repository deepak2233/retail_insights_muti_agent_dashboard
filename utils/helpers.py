"""
Utility functions for Retail Insights Assistant
"""
import json
from datetime import datetime
from typing import Any, Dict


def format_currency(amount: float) -> str:
    """Format number as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format number as percentage"""
    return f"{value:.1f}%"


def format_number(value: float) -> str:
    """Format number with thousands separator"""
    return f"{value:,.0f}"


def save_conversation(conversation_history: list, filename: str = None):
    """Save conversation history to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(conversation_history, f, indent=2)
    
    return filename


def load_conversation(filename: str) -> list:
    """Load conversation history from file"""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_growth(current: float, previous: float) -> float:
    """Calculate percentage growth"""
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_date_range(start_date: str, end_date: str) -> str:
    """Format date range nicely"""
    return f"{start_date} to {end_date}"
