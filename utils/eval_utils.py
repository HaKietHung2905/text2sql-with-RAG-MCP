"""
Utility functions for evaluation.
"""

import re
from typing import Optional


def normalize_sql_for_evaluation(sql: Optional[str]) -> Optional[str]:
    """
    Normalize SQL query for fair comparison
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL query
    """
    if not sql:
        return sql
    
    # Remove newlines and normalize whitespace
    normalized = ' '.join(sql.strip().split())
    
    # Remove trailing semicolon
    if normalized.endswith(';'):
        normalized = normalized[:-1].strip()  # Strip after removing semicolon
    
    return normalized


def extract_db_name_from_question(line: str) -> Optional[tuple]:
    """
    Extract question and database name from question line
    
    Args:
        line: Question line in format "question\tdb_name"
        
    Returns:
        Tuple of (question, db_name) or None
    """
    parts = None
    
    if '\t' in line:
        parts = line.split('\t')
    elif '  ' in line:
        parts = re.split(r'\s{2,}', line)
    elif '\\t' in line:
        parts = line.split('\\t')
    elif ' ' in line:
        parts = line.rsplit(' ', 1)
    
    if not parts or len(parts) < 2:
        return None
    
    question = parts[0].strip()
    db_name = parts[1].strip()
    
    return question, db_name


def clean_sql_string(sql: str) -> str:
    """
    Clean SQL string by removing backticks and normalizing
    
    Args:
        sql: SQL string
        
    Returns:
        Cleaned SQL string
    """
    if isinstance(sql, str):
        sql = sql.strip('`').replace('`', '')
    return sql