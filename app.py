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


st.sidebar.header("API Key")

st.sidebar.caption(
    "Your API key is stored only for this browser session and is never saved."
)

st.sidebar.text_input(
    "OpenAI API key",
    type="password",
    key="user_api_key"
)

def clear_api_key() -> None:
    st.session_state["user_api_key"] = ""

st.sidebar.button("Clear key", on_click=clear_api_key)


def get_active_api_key() -> str | None:
    key = st.session_state.get("user_api_key", "").strip()
    return key if key else None


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

if not get_active_api_key():
    st.warning("Add your OpenAI API key in the sidebar to run analysis.")
    st.stop()


# -----------------------------
# LLM Prompt Loading
# -----------------------------

# Keeps the prompt external so it can be edited
# without touching application logic
def load_prompt() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()

def load_step_prompt(filename: str, **kwargs: Any) -> str:
    """
    Loads a step prompt from /prompts and formats it with provided kwargs.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(script_dir, "prompts")
    prompt_path = os.path.join(prompts_dir, filename)
    with open(prompt_path, "r") as f:
        template = f.read()
    return template.format(**kwargs)

# -----------------------------
# LLM Interaction and Response Handling
# -----------------------------

# Load environment variables from .env before reading API key
load_dotenv()

MASTER_PROMPT = load_prompt()

# Initialize OpenAI client with error checking
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Server configuration error: API key not found.")
    st.stop()

from openai import OpenAI

def get_client() -> OpenAI:
    key = get_active_api_key()
    if not key:
        raise ValueError("No API key provided.")
    return OpenAI(api_key=key)



# This function calls OpenAI's GPT-4 model
def call_llm(prompt: str, temperature: float = 0.2) -> str:
    """
    Sends the full prompt to the LLM using the user's API key
    and returns a JSON string.
    """
    client = get_client()  # per-session client

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # cheaper + stable for chaining
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert negotiation coach who analyzes emails structurally.

CRITICAL INSTRUCTIONS:
- Do NOT draft or generate email replies
- Do NOT suggest specific wording
- Do NOT take sides
- Do NOT recommend accepting or rejecting
- Output ONLY valid JSON matching the exact schema provided
- Include no text before or after the JSON
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=2000
        )

        result = response.choices[0].message.content

        if not result or not result.strip():
            raise ValueError("Empty response from API")

        return result.strip()

    except Exception as e:
        st.error("LLM call failed.")
        st.exception(e)
        raise

# -----------------------------
# JSON Parsing Function
# -----------------------------


def call_llm_json(prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    raw = call_llm(prompt, temperature=temperature)
    print("RAW LLM RESPONSE:", raw)
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return json.loads(cleaned)

#-----------------------------
# Step A: Extracting Key Elements
#-----------------------------
# Extracts key elements from the LLM output


def call_llm_json_with_retry(prompt: str) -> Dict[str, Any]:
    try:
        return call_llm_json(prompt, temperature=0.2)
    except Exception as e:
        print("Step failed at temp 0.2, retrying at 0.0")
        return call_llm_json(prompt, temperature=0.0)

def validate_step_a(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Step A output must be an object")
    if "facts" not in data or "evidence" not in data:
        raise ValueError("Step A missing required keys: facts, evidence")
    facts = data["facts"]
    if not isinstance(facts, dict):
        raise ValueError("Step A facts must be an object")
    required_fact_keys = {
        "explicit_asks", "deadlines", "numbers_and_terms",
        "constraints", "authority_cues", "ambiguities"
    }
    if not required_fact_keys.issubset(facts.keys()):
        raise ValueError("Step A facts missing required fields")
    for k in required_fact_keys:
        if not isinstance(facts[k], list):
            raise ValueError(f"Step A facts.{k} must be a list")
    if not isinstance(data["evidence"], list):
        raise ValueError("Step A evidence must be a list")
    for ev in data["evidence"]:
        if not isinstance(ev, dict) or "claim" not in ev or "quote" not in ev:
            raise ValueError("Step A evidence entries must have claim and quote")

def validate_step_b(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Step B output must be an object")
    if "email_diagnosis" not in data or "checklist_gaps" not in data:
        raise ValueError("Step B missing required keys: email_diagnosis, checklist_gaps")
    diag = data["email_diagnosis"]
    if not isinstance(diag, dict):
        raise ValueError("Step B email_diagnosis must be an object")
    required_diag_keys = {
        "negotiation_type", "negotiation_stage", "power_signals",
        "emotional_tone", "uncertainty_flags"
    }
    if not required_diag_keys.issubset(diag.keys()):
        raise ValueError("Step B email_diagnosis missing required fields")
    if not isinstance(diag["power_signals"], dict):
        raise ValueError("Step B power_signals must be an object")
    for k in {"deadlines", "alternatives", "authority"}:
        if k not in diag["power_signals"]:
            raise ValueError("Step B power_signals missing required fields")
    if not isinstance(diag["emotional_tone"], dict):
        raise ValueError("Step B emotional_tone must be an object")
    for k in {"primary", "evidence"}:
        if k not in diag["emotional_tone"]:
            raise ValueError("Step B emotional_tone missing required fields")
    if not isinstance(diag["uncertainty_flags"], list):
        raise ValueError("Step B uncertainty_flags must be a list")
    if not isinstance(data["checklist_gaps"], list):
        raise ValueError("Step B checklist_gaps must be a list")
    for gap in data["checklist_gaps"]:
        if not isinstance(gap, dict):
            raise ValueError("Each checklist gap must be an object")
        for k in {"item", "why_it_matters", "theory_reference"}:
            if k not in gap:
                raise ValueError("Checklist gap missing required fields")

def validate_step_c(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Step C output must be an object")
    if "coaching" not in data or "missing_context_questions" not in data:
        raise ValueError("Step C missing required keys: coaching, missing_context_questions")
    coaching = data["coaching"]
    if not isinstance(coaching, dict):
        raise ValueError("Step C coaching must be an object")
    for k in {"reflection_questions", "risk_watch_outs", "value_levers"}:
        if k not in coaching or not isinstance(coaching[k], list):
            raise ValueError(f"Step C coaching.{k} must be a list")
    if not isinstance(data["missing_context_questions"], list):
        raise ValueError("Step C missing_context_questions must be a list")

def step_a_extract(email: str, context: str) -> Dict[str, Any]:
    prompt_a = load_step_prompt("step_a.txt", email=email, context=context)
    out_a = call_llm_json_with_retry(prompt_a)
    print("RAW STEP A OUTPUT:", out_a)
    
    validate_step_a(out_a)
    return out_a

def step_b_diagnose(extracted: Dict[str, Any], context: str) -> Dict[str, Any]:
    prompt_b = load_step_prompt("step_b.txt", extracted_json=json.dumps(extracted), context=context)
    out_b = call_llm_json_with_retry(prompt_b)
    validate_step_b(out_b)
    return out_b

def step_c_coach(diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    prompt_c = load_step_prompt("step_c.txt", diagnosis_json=json.dumps(diagnosis))
    out_c = call_llm_json_with_retry(prompt_c)
    validate_step_c(out_c)
    return out_c

def analyze_email_v2(email: str, context: str) -> Dict[str, Any]:
    a = step_a_extract(email, context)
    b = step_b_diagnose(a, context)
    c = step_c_coach(b)

    return {
        "step_a": a,
        "step_b": b,
        "step_c": c
    }




# -----------------------------
# Analyze Button Logic
# -----------------------------
if run_tests:
    test_cases = load_test_emails()
    for case in test_cases:
        st.markdown(f"### Test Case: {case['id']}")
        try:
            result = analyze_email_v2(case["email_text"], case["context"])
            st.success("Chain ran successfully")

            diag = result["step_b"]["email_diagnosis"]
            st.markdown(f"- Type: {diag['negotiation_type']}")
            st.markdown(f"- Stage: {diag['negotiation_stage']}")
            st.markdown(f"- Tone: {diag['emotional_tone']['primary']}")

            st.markdown("Checklist gaps:")
            for gap in result["step_b"]["checklist_gaps"]:
                st.markdown(f"- {gap['item']}")

            st.markdown("Coaching outputs:")
            st.markdown(f"- Reflection questions: {len(result['step_c']['coaching']['reflection_questions'])}")
            st.markdown(f"- Risk watch outs: {len(result['step_c']['coaching']['risk_watch_outs'])}")
            st.markdown(f"- Value levers: {len(result['step_c']['coaching']['value_levers'])}")

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
                result = analyze_email_v2(email_text, context_text)
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

    
    st.subheader("Panel 1: Facts and Signals")
    st.info(
    "This email contains a time-bound request with implied alternatives if no response is received."
    )
    facts = result["step_a"]["facts"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Explicit Asks**")
        if facts["explicit_asks"]:
            for item in facts["explicit_asks"]:
                st.markdown(f"- {item}")
        else:
            st.markdown("_No explicit asks detected_")

        st.markdown("**Deadlines**")
        st.write(facts["deadlines"])
        st.markdown("**Numbers and Terms**")
        
        if facts["numbers_and_terms"]:
            st.markdown("**Numbers and Terms**")
            for n in facts["numbers_and_terms"]:
                st.markdown(f"- {n}")


    with col2:
        st.markdown("**Constraints**")
        st.write(facts["constraints"])
        st.markdown("**Authority Cues**")
        st.write(facts["authority_cues"])
        st.markdown("**Ambiguities**")
        st.write(facts["ambiguities"])

    st.markdown("**Evidence Quotes**")
    for ev in result["step_a"]["evidence"]:
        st.markdown(f"- **{ev['claim']}**: \"{ev['quote']}\"")

    with st.expander("See supporting evidence"):
        for ev in result["step_a"]["evidence"]:
            st.markdown(f"**{ev['claim']}**")
            st.markdown(f"> {ev['quote']}")

    st.subheader("Panel 2: Diagnosis and Checklist Gaps")
    diag = result["step_b"]["email_diagnosis"]
    st.markdown(f"**Negotiation Type:** {diag['negotiation_type']}")
    st.markdown(f"**Negotiation Stage:** {diag['negotiation_stage']}")

    st.markdown("**Power Signals**")
    st.write(diag["power_signals"])

    st.markdown("**Emotional Tone**")
    st.markdown(f"- Primary: {diag['emotional_tone']['primary']}")
    st.markdown(f"- Evidence: {diag['emotional_tone']['evidence']}")

    if diag["uncertainty_flags"]:
        st.markdown("**Uncertainty Flags**")
        st.write(diag["uncertainty_flags"])

    st.markdown("**Checklist Gaps**")
    gaps = result["step_b"]["checklist_gaps"]
    if not gaps:
        st.info("No major preparation gaps detected based on the provided context.")
    else:
        for gap in gaps:
            st.markdown(f"**{gap['item']}**")
            st.markdown(f"- Why it matters: {gap['why_it_matters']}")
            st.markdown(f"- Theory reference: {gap['theory_reference']}")

    st.subheader("Panel 3: Coaching Prompts")
    coaching = result["step_c"]["coaching"]

    st.markdown("**Reflection Questions**")
    for q in coaching["reflection_questions"]:
        st.markdown(f"- {q}")

    st.markdown("**Risk Watch Outs**")
    for w in coaching["risk_watch_outs"]:
        st.markdown(f"- {w}")

    st.markdown("**Value Levers to Consider**")
    for v in coaching["value_levers"]:
        st.markdown(f"- {v}")

    if result["step_c"]["missing_context_questions"]:
        st.markdown("**Missing Context Questions**")
        for q in result["step_c"]["missing_context_questions"]:
            st.markdown(f"- {q}")
