import hashlib
import json
import logging
import re
import os
import pandas as pd
import database
import llm_engine
import requests  # Ensure you've run 'pip install requests'

# --- CONFIGURATION ---
# Your unique Slack URL - use environment variable
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "please", "the", "to", "was", "what", "when", "where", "with", "you", "your"}

# --- MENTOR LOGIC HELPERS ---

def confidence_label(score):
    """Mentor logic: Categorizes the AI's confidence and returns CSS classes"""
    if score >= 0.75:
        return "High Confidence", "gap-high"
    elif score >= 0.5:
        return "Medium Confidence", "gap-mid"
    return "Low Confidence", "gap-low"

def normalize_markdown(text):
    """Mentor logic: Cleans up AI resolution text indents and bullets"""
    if not text: return ""
    normalized_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith(("* ", "â€¢ ", "• ")):
            normalized_lines.append(f"{indent} • {stripped[2:].lstrip()}")
        elif re.match(r"^\d+\)\s+", stripped):
            normalized_lines.append(f"{indent}{re.sub(r'^(\d+)\)\s+', r'\1. ', stripped)}")
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()

# --- SLACK INTEGRATION ---

def send_slack_alert(ticket_title, category, priority, confidence):
    """Sends a professional, color-coded alert to Slack channel"""
    if confidence < 0.4:
        color = "#ef4444"  # Red for Knowledge Gaps
        status_text = "⚠️ Knowledge Gap Detected"
    elif confidence < 0.7:
        color = "#f59e0b"  # Orange for Tentative
        status_text = "⚡ Tentative AI Resolution"
    else:
        color = "#22c55e"  # Green for High Confidence
        status_text = "✅ High Confidence Resolution"

    payload = {
        "attachments": [
            {
                "color": color,
                "pretext": f"*{status_text}*",
                "title": f"New Ticket: {ticket_title}",
                "fields": [
                    {"title": "Category", "value": category, "short": True},
                    {"title": "Priority", "value": priority, "short": True},
                    {"title": "AI Confidence", "value": f"{round(confidence * 100)}%", "short": True}
                ],
                "footer": "RAG Intelligence Bot",
                "footer_icon": "https://cdn-icons-png.flaticon.com/512/2111/2111615.png"
            }
        ]
    }
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"Slack Alert Error: {e}")

# --- DATABASE OPERATIONS ---

def normalize_ticket_text(title, description):
    tokens = re.sub(r"[^a-z0-9\s]", " ", (title + " " + description).lower()).split()
    tokens = [t for t in tokens if t and t not in STOP_WORDS]
    return " ".join(tokens[:6]).strip() or "general support request"

def submit_ticket(title, description, category, priority, user_id):
    """Processes a new ticket, saves to DB, and sends Slack alert"""
    # 1. AI Analysis
    analysis = llm_engine.analyze_ticket(title, description, priority, category)
    normalized_query = normalize_ticket_text(title, description)
    
    # 2. Database Insertion
    conn = database.get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tickets (title, description, category, priority, user_id, ai_resolution,
            confidence_score, resolution_status, retrieval_score, kb_context_found, normalized_query)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (title, description, category, priority, user_id, analysis["resolution_text"],
             analysis["confidence_score"], analysis["resolution_status"], analysis["retrieval_score"],
             int(bool(analysis["kb_context_found"])), normalized_query))
        conn.commit()
    finally:
        conn.close()

    # 3. Trigger Slack Alert
    send_slack_alert(title, category, priority, analysis["confidence_score"])
    
    return analysis

def get_admin_kpis():
    conn = database.get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) AS total_tickets,
            SUM(CASE WHEN resolution_status IN ('resolved', 'tentative') THEN 1 ELSE 0 END) AS resolved_tickets,
            AVG(confidence_score) AS avg_confidence,
            SUM(CASE WHEN feedback_value = 'helpful' THEN 1 ELSE 0 END) AS helpful_count,
            SUM(CASE WHEN feedback_value = 'not_helpful' THEN 1 ELSE 0 END) AS not_helpful_count
            FROM tickets""")
        res = cursor.fetchone()
        row = dict(res) if res else {}
        helpful = row.get("helpful_count") or 0
        not_helpful = row.get("not_helpful_count") or 0
        total_fb = helpful + not_helpful
        return {
            "total_tickets": row.get("total_tickets") or 0,
            "resolved_tickets": row.get("resolved_tickets") or 0,
            "avg_confidence": round(row.get("avg_confidence") or 0.0, 3),
            "helpful_count": helpful,
            "not_helpful_count": not_helpful,
            "helpful_rate": round(helpful / total_fb, 3) if total_fb > 0 else 0.0
        }
    finally:
        conn.close()

def get_analytics_data():
    conn = database.get_db_connection()
    try:
        df = pd.read_sql_query("SELECT category, AVG(confidence_score) as conf FROM tickets GROUP BY category", conn)
        labels = df['category'].tolist()
        values = [round(x, 2) for x in df['conf'].tolist()]
        gaps = pd.read_sql_query("""
            SELECT category, normalized_query as topic, AVG(confidence_score) as conf, COUNT(*) as count 
            FROM tickets GROUP BY category, topic ORDER BY conf ASC LIMIT 5
        """, conn).to_dict('records')
        return labels, values, gaps
    finally:
        conn.close()

def get_user_tickets(user_id):
    conn = database.get_db_connection()
    try:
        query = "SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC"
        return pd.read_sql_query(query, conn, params=(user_id,))
    finally:
        conn.close()