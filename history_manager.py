"""
history_manager.py
SQLite-based persistence manager for the AI College FAQ Chatbot.
Handles conversation histories, session management, and student query analytics.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import DB_FILE

def get_db_connection():
    """Establish connection to SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables for sessions, messages, and query analytics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources_json TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)
    
    # Analytics table to track FAQ topics and performance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query TEXT NOT NULL,
            category TEXT,
            latency_seconds REAL,
            status TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def ensure_session(session_id: str, title: str = "New Student Inquiry"):
    """Creates a session if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO sessions (session_id, title) VALUES (?, ?)", (session_id, title))
        conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None):
    """Saves a message to the SQLite database."""
    ensure_session(session_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    sources_str = json.dumps(sources) if sources else None
    cursor.execute("""
        INSERT INTO messages (session_id, role, content, sources_json)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, sources_str))
    conn.commit()
    conn.close()

def get_messages(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves all chat messages for a specific session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, sources_json, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        sources = json.loads(row["sources_json"]) if row["sources_json"] else []
        messages.append({
            "role": row["role"],
            "content": row["content"],
            "sources": sources,
            "timestamp": row["timestamp"]
        })
    return messages

def clear_session_messages(session_id: str):
    """Deletes all messages for a session while retaining the session record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def log_analytics(session_id: str, query: str, category: str, latency: float, status: str = "SUCCESS"):
    """Logs query metadata for administrator reporting and FAQ trend analysis."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO query_analytics (session_id, query, category, latency_seconds, status)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, query, category, round(latency, 3), status))
    conn.commit()
    conn.close()

def get_analytics_summary() -> Dict[str, Any]:
    """Generates an analytics summary for the administrative dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total_queries FROM query_analytics")
    total_queries = cursor.fetchone()["total_queries"]
    
    cursor.execute("SELECT AVG(latency_seconds) as avg_latency FROM query_analytics WHERE status='SUCCESS'")
    avg_latency_row = cursor.fetchone()
    avg_latency = round(avg_latency_row["avg_latency"], 2) if avg_latency_row and avg_latency_row["avg_latency"] else 0.0
    
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM query_analytics
        GROUP BY category
        ORDER BY count DESC
    """)
    category_rows = cursor.fetchall()
    categories = {row["category"]: row["count"] for row in category_rows}
    
    cursor.execute("""
        SELECT query, category, latency_seconds, status, timestamp
        FROM query_analytics
        ORDER BY id DESC
        LIMIT 10
    """)
    recent_queries = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return {
        "total_queries": total_queries,
        "avg_latency": avg_latency,
        "categories": categories,
        "recent_queries": recent_queries
    }

# Ensure database tables exist on module import
init_db()
