import ollama
import logging
import re
import config
import rag_engine
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "llama3.2:1b"
GENERIC_RESPONSE_PATTERNS = (
    "contact support",
    "reach out to your administrator",
    "not enough information",
    "unable to determine",
    "please provide more details",
)

def check_model_availability():
    """Checks if the model is available locally, pulls if not."""
    try:
        models_response = ollama.list()
        model_names = []
        if 'models' in models_response:
            for m in models_response['models']:
                if isinstance(m, dict):
                    model_names.append(m.get('name', ''))
                    model_names.append(m.get('model', '')) 

        if MODEL_NAME not in model_names and f"{MODEL_NAME}:latest" not in model_names:
            logging.info(f"Model {MODEL_NAME} not found. Pulling...")
            ollama.pull(MODEL_NAME)
            logging.info(f"Model {MODEL_NAME} pulled successfully.")
        else:
            logging.info(f"Model {MODEL_NAME} is ready.")
    except Exception as e:
        logging.warning(f"Error checking model list ({e}). Attempting pull to be safe...")
        try:
            ollama.pull(MODEL_NAME)
        except Exception as pull_error:
            logging.error(f"Failed to pull model: {pull_error}")

def _slugify_filename(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(tokens[:6]) if tokens else "missing_knowledge_article"

def _suggest_kb_filename(title, description, category):
    source_text = f"{title} {description}".strip().lower()
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", source_text)
        if token not in {"the", "and", "for", "with", "from", "that", "this", "have", "need", "cannot", "cant"}
    ]
    phrase = " ".join(tokens[:6]) or category or "knowledge gap"
    return f"{_slugify_filename(phrase)}_guide.md"

def _calculate_confidence(retrieval_score, kb_context_found, resolution_text, had_error):
    """
    DYNAMIC SCORING LOGIC:
    Instead of fixed jumps, it uses the RAG retrieval score as a base.
    """
    if had_error:
        return 0.0

    # 1. Base Score from Retrieval (ranges roughly 0.1 to 0.7)
    # We add a small random jitter to ensure 10% vs 12% vs 15% variety
    confidence = retrieval_score + random.uniform(-0.02, 0.05)

    # 2. Context Bonus
    if kb_context_found:
        # If we found matches in PDFs, boost significantly
        confidence += 0.25
    else:
        # If no PDF context, it's just a general guess
        confidence = min(confidence, 0.35)

    # 3. Quality Check
    if resolution_text and len(resolution_text.strip()) >= 150:
        confidence += 0.1  # Thorough answers get a small boost
    
    # 4. Penalty for "I don't know" responses
    lowered = resolution_text.lower()
    if any(pattern in lowered for pattern in GENERIC_RESPONSE_PATTERNS):
        confidence = 0.10  # Hard floor for failures

    # Final rounding to show dynamic percentages (e.g., 0.672 -> 67.2%)
    return max(0.1, min(0.98, round(confidence, 3)))

def analyze_ticket(title, description, priority, category):
    """
    Uses the LLM to generate a resolution and AI quality metadata.
    """
    logging.info("Retrieving relevant context from RAG Engine...")
    
    # This calls your rag_engine which searches the PDFs
    retrieval = rag_engine.get_relevant_context(f"{title} {description}")
    
    context = retrieval.get("context_text", "")
    retrieval_score = retrieval.get("retrieval_score", 0.0)
    kb_context_found = retrieval.get("kb_context_found", False)

    prompt = f"""
    Context from Knowledge Base:
    {context}
    
    User Ticket: {title}
    Details: {description}
    
    Instruction:
    Provide a professional technical resolution based ONLY on the context above. 
    If context is missing, use general expertise but keep the confidence score low.
    - Be concise and use bullet points.
    - Do NOT mention being an AI.
    
    Resolution:
    """

    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        resolution_text = response['message']['content'].strip()
        
        # Calculate the dynamic score
        confidence_score = _calculate_confidence(
            retrieval_score=retrieval_score,
            kb_context_found=kb_context_found,
            resolution_text=resolution_text,
            had_error=False,
        )

        resolved_threshold = config.get_float_env("AI_CONFIDENCE_THRESHOLD", 0.60)
        resolution_status = (
            "resolved" if confidence_score >= resolved_threshold else "tentative"
        )

        return {
            "category": category,
            "resolution_text": resolution_text,
            "confidence_score": confidence_score,
            "resolution_status": resolution_status,
            "retrieval_score": retrieval_score,
            "kb_context_found": kb_context_found,
            "context_matches": retrieval.get("matches", []),
            "suggested_kb_filename": (
                None if resolution_status == "resolved"
                else _suggest_kb_filename(title, description, category)
            ),
            "error": None,
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logging.error(f"LLM Error: {error_msg}")
        return {
            "category": category,
            "resolution_text": f"Failed to generate resolution. Details: {error_msg}",
            "confidence_score": 0.0,
            "resolution_status": "unresolved",
            "retrieval_score": 0.0,
            "kb_context_found": False,
            "context_matches": [],
            "suggested_kb_filename": _suggest_kb_filename(title, description, category),
            "error": error_msg,
        }

if __name__ == "__main__":
    check_model_availability()
    # Test run
    analysis = analyze_ticket("Network Protocol Failure", "Unable to catch the connection for the wi-fi", "High", "Network")
    print(f"Confidence: {analysis['confidence_score'] * 100}%")
    print(f"Status: {analysis['resolution_status']}")