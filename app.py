# -----------------------------
# Imports
# -----------------------------
# streamlit: UI framework for quick interactive apps
# json: parsing structured LLM output
# typing: improves clarity for validation functions
import streamlit as st
import json
from typing import Any, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI

def load_test_emails():
    """Load test email cases from JSON file."""
    try:
        # Use absolute path based on current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "test_emails.json")
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("test_emails.json not found")
        return []
    except json.JSONDecodeError:
        st.error("test_emails.json contains invalid JSON")
        return []
    


# -----------------------------
# Streamlit App Configuration
# -----------------------------

# Sets page title and layout for the app
st.set_page_config(
    page_title="Email Negotiation Coach",
    layout="centered"
)

# App title and positioning message
st.title("Email Negotiation Coach")
st.caption("Reflection-first analysis. No replies are drafted.")

# -----------------------------
# User Input Section
# -----------------------------

# Primary input: the incoming negotiation email
email_text = st.text_area(
    "Incoming negotiation email",
    height=220,
    placeholder="Paste the email you received here"
)

# Optional context to improve analysis quality
context_text = st.text_area(
    "Optional context (stakes, relationship importance)",
    height=80,
    placeholder="Example: High stakes, long-term relationship"
)

run_tests = st.checkbox("Run test suite instead of manual input")

# -----------------------------
# LLM Prompt Loading
# -----------------------------

# Keeps the prompt external so it can be edited
# without touching application logic
def load_prompt() -> str:
    with open("prompt.txt", "r") as f:
        return f.read()

MASTER_PROMPT = load_prompt()

# -----------------------------
# LLM Interaction and Response Handling
# -----------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Server configuration error: API key not found.")
    st.stop()

# Load environment variables from .env file
load_dotenv()

MASTER_PROMPT = load_prompt()

# Initialize OpenAI client with error checking
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")

client = OpenAI(api_key=api_key)

# This function calls OpenAI's GPT-4 model
def call_llm(prompt: str) -> str:
    """
    Sends the full prompt to OpenAI's GPT-4 and returns the response as JSON.
    
    Args:
        prompt: The complete prompt including instructions and user input
    
    Returns:
        A string containing valid JSON output from the LLM
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Use gpt-4-turbo for better instruction following
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert negotiation coach who analyzes emails structurally.

CRITICAL INSTRUCTIONS:
- Do NOT draft or generate email replies
- Do NOT suggest specific wording
- Do NOT take sides
- Output ONLY valid JSON matching the exact schema provided
- Include no text before or after the JSON

Your output must be a JSON object with exactly these top-level keys:
- "email_diagnosis" (object with negotiation_type, negotiation_stage, power_signals, emotional_tone)
- "checklist_gaps" (array of gap objects)

Nothing else."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,  # Very low for strict adherence to instructions
            max_tokens=2000
        )
        
        # Extract the text response from the API
        result = response.choices[0].message.content
        
        if not result:
            raise ValueError("Empty response from API")
        
        return result
        
    except Exception as e:
        st.error(f"LLM API Error: {str(e)}")
        raise

# -----------------------------
# Schema Validation Constants
# -----------------------------
# Define required keys for validation

REQUIRED_TOP_KEYS = {"email_diagnosis", "checklist_gaps"}

REQUIRED_DIAG_KEYS = {
    "negotiation_type",
    "negotiation_stage",
    "power_signals",
    "emotional_tone"
}

REQUIRED_POWER_KEYS = {"deadlines", "alternatives", "authority"}

REQUIRED_TONE_KEYS = {"primary", "evidence"}

REQUIRED_GAP_KEYS = {"item", "why_it_matters", "theory_reference"}

# -----------------------------
# Schema Validation Function
# -----------------------------
# Ensures the LLM output matches the expected structure
def validate_schema(data: Dict[str, Any]) -> None:

    # Debug: print actual keys
    print(f"Actual keys in response: {data.keys()}")
    print(f"Expected keys: {REQUIRED_TOP_KEYS}")
    
    # Check top-level structure
    if not REQUIRED_TOP_KEYS.issubset(data.keys()):
        st.error(f"Response has keys: {list(data.keys())}")
        st.error(f"Expected keys: {REQUIRED_TOP_KEYS}")
        raise ValueError("Missing top-level keys in output")

    diag = data["email_diagnosis"]

    # Validate diagnosis section
    if not REQUIRED_DIAG_KEYS.issubset(diag.keys()):
        raise ValueError("Missing fields in email_diagnosis")

    # Validate nested power signals (can be dict, list, or string)
    if not isinstance(diag["power_signals"], (dict, list, str)):
        raise ValueError("power_signals must be a dict, list, or string")

    # Validate emotional tone structure (can be dict or string)
    if isinstance(diag["emotional_tone"], dict):
        if not REQUIRED_TONE_KEYS.issubset(diag["emotional_tone"].keys()):
            raise ValueError("Missing emotional tone fields")
    elif not isinstance(diag["emotional_tone"], str):
        raise ValueError("emotional_tone must be a dict or string")

    # Validate checklist gaps list
    if not isinstance(data["checklist_gaps"], list):
        raise ValueError("checklist_gaps must be a list")

    # Just check that gaps are dicts (don't enforce exact keys)
    for gap in data["checklist_gaps"]:
        if not isinstance(gap, dict):
            raise ValueError("Each gap must be a dict")

# -----------------------------
# Core Analysis Function
# -----------------------------
# Combines user input with the master prompt,
# calls the LLM, parses JSON, and validates schema
def analyze_email(email: str, context: str) -> Dict[str, Any]:

    # Construct the full prompt sent to the LLM
    full_prompt = f"""
{MASTER_PROMPT}

Incoming Email:
{email}

Optional Context:
{context}
"""

    # Call the language model
    raw_response = call_llm(full_prompt)
    
    # Strip markdown code block formatting if present
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:]  # Remove ```json
    if raw_response.startswith("```"):
        raw_response = raw_response[3:]  # Remove ```
    
    raw_response = raw_response.strip()
    
    if raw_response.endswith("```"):
        raw_response = raw_response[:-3]  # Remove trailing ```
    
    raw_response = raw_response.strip()
    
    # Debug: print raw response for troubleshooting
    print(f"Cleaned LLM Response: {raw_response[:300]}...")
    
    # Parse JSON output with error handling
    try:
        parsed = json.loads(raw_response)
        print(f"Parsed JSON keys: {parsed.keys()}")
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse LLM response as JSON: {e}")
        st.error(f"Response was: {raw_response[:500]}")
        raise

    # Validate structure before returning
    validate_schema(parsed)

    return parsed

# -----------------------------
# Analyze Button Logic
# -----------------------------
if run_tests:
    test_cases = load_test_emails()
    for case in test_cases:
        st.markdown(f"### Test Case: {case['id']}")
        try:
            result = analyze_email(case["email_text"], case["context"])
            st.success("Schema valid")

            diag = result["email_diagnosis"]

            st.markdown(f"- Type: {diag['negotiation_type']}")
            st.markdown(f"- Stage: {diag['negotiation_stage']}")
            
            # Handle emotional_tone as string or dict
            if isinstance(diag['emotional_tone'], dict):
                tone = diag['emotional_tone'].get('primary', 'Unknown')
            else:
                tone = diag['emotional_tone']
            st.markdown(f"- Tone: {tone}")

            st.markdown("Checklist gaps:")
            for gap in result["checklist_gaps"]:
                # Handle gap as dict with various possible keys
                gap_label = gap.get('item') or gap.get('gap') or str(gap)
                st.markdown(f"- {gap_label}")

        except Exception as e:
            st.error("Test failed")
            st.exception(e)

# Handles user interaction and error safety
if st.button("Analyze Email"):

    # Prevent empty submissions
    if not email_text.strip():
        st.error("Please paste an email to analyze.")

    else:
        with st.spinner("Analyzing negotiation context..."):
            try:
                result = analyze_email(email_text, context_text)
                st.success("Analysis complete")

            except Exception as e:
                # Fail loudly so errors are visible during demo
                st.error("Analysis failed.")
                st.exception(e)

# -----------------------------
# Render Diagnosis Output
# -----------------------------

# Displays structured negotiation diagnosis
if "result" in locals():

    st.subheader("Email Diagnosis Panel")

    diag = result["email_diagnosis"]

    # High level classification
    st.markdown(f"""
**Negotiation Type:** {diag['negotiation_type']}  
**Negotiation Stage:** {diag['negotiation_stage']}
""")

    # Power signals section
    st.markdown("**Power Signals**")
    signals = diag['power_signals']
    if isinstance(signals, dict):
        signals_text = "\n".join([f"- {k}: {v}" for k, v in signals.items()])
    elif isinstance(signals, list):
        signals_text = "\n".join([f"- {s}" for s in signals])
    else:
        signals_text = f"- {signals}"
    st.markdown(signals_text)

    # Emotional tone section
    st.markdown("**Emotional Tone**")
    tone = diag['emotional_tone']
    if isinstance(tone, dict):
        tone_text = "\n".join([f"- {k}: {v}" for k, v in tone.items()])
    else:
        tone_text = f"- {tone}"
    st.markdown(tone_text)
    
    # -----------------------------
    # Render Checklist Gap Detector
    # -----------------------------
    st.subheader("Checklist Gap Detector")

    gaps = result["checklist_gaps"]

    # If no gaps are found, make that explicit
    if not gaps:
        st.info("No major preparation gaps detected based on the provided context.")

    # Otherwise, list each gap with explanation
    else:
        for gap in gaps:
            # Handle flexible key names
            item = gap.get('item') or gap.get('gap') or gap.get('missing') or str(gap)
            why = gap.get('why_it_matters') or gap.get('explanation') or gap.get('reason') or ''
            theory = gap.get('theory_reference') or gap.get('reference') or gap.get('concept') or ''
            
            st.markdown(f"""
**{item}**  
Why it matters: {why}  
Theory reference: {theory}
""")
        
# -----------------------------
# Schema Definitions
# -----------------------------
# Required keys for top-level JSON structure
REQUIRED_TOP_KEYS = {"email_diagnosis", "checklist_gaps"}

# Required fields inside diagnosis
REQUIRED_DIAG_KEYS = {
    "negotiation_type",
    "negotiation_stage",
    "power_signals",
    "emotional_tone"
}

# Required fields for nested objects
REQUIRED_POWER_KEYS = {"deadlines", "alternatives", "authority"}
REQUIRED_TONE_KEYS = {"primary", "evidence"}
REQUIRED_GAP_KEYS = {"item", "why_it_matters", "theory_reference"}