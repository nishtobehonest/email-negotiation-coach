# V2 - Email Negotiation Coach

A reflection-first negotiation coaching tool designed to help users slow down, spot blind spots, and think clearly before replying to negotiation emails.

This project is intentionally human-in-the-loop. It does not draft replies or recommend actions. Instead, it helps users understand what is happening in a negotiation and asks better questions before they respond.

---

## Why this project

In negotiation classes and real life, people often react emotionally or instinctively to emails, especially under time pressure. While frameworks like the negotiation checklist are taught, they are rarely applied in real time.

This tool bridges that gap by applying negotiation concepts live, in a structured and reflective way.

---

## What the app does

**Given:**
- An incoming negotiation email  
- Optional context such as stakes, relationship importance, or constraints  

**The app provides:**
- A structured diagnosis of the negotiation context  
- Identification of preparation gaps using the negotiation checklist  
- Coaching-style reflection prompts to help the user pause and think  
- Risk watch-outs and value levers to consider  

**The app never:**
- Drafts email replies  
- Suggests wording  
- Recommends accepting or rejecting an offer  
- Takes sides in the negotiation  

---

## Design principles

This project is guided by a few core principles:

- **Reflection over reaction**  
  The goal is to slow down thinking, not speed up replies.

- **Coaching, not automation**  
  The tool surfaces blind spots and questions instead of answers.

- **Evidence-based reasoning**  
  Every insight is grounded in specific language from the email.

- **Human judgment preserved**  
  The final decision always stays with the user.

---

## Version 2 direction (current)

Version 1 focused on classification and checklist diagnosis. Version 2 expands this into a coaching experience and addresses scalability and usability concerns.

### Key improvements in v2

#### 1. User-provided API keys (secure)
- Users can enter their own OpenAI API key via the sidebar  
- Keys are stored only in Streamlit session state  
- Keys are never written to disk or logs  
- Clearing the session clears the key  
- This avoids shared costs and protects user privacy  

#### 2. Multi-step reasoning chain (not a single LLM call)

The app now runs as a structured reasoning process:

**Step A: Extract facts and signals**
- Explicit asks  
- Deadlines  
- Numbers and constraints  
- Authority cues  
- Ambiguities  
- Quoted evidence from the email  

**Step B: Diagnose and checklist gaps**
- Negotiation type and stage  
- Power signals and emotional tone  
- Missing preparation items  
- Uncertainty flags when key information is missing  

**Step C: Coaching layer**
- Reflection questions a coach would ask  
- Risk watch-outs phrased as things to be careful about  
- Value levers to consider without advice or wording  

This chain makes the app feel like a thoughtful coach rather than a classifier.

---

## Safety and constraints

These constraints are intentional and non-negotiable:

- No reply drafting  
- No suggested wording  
- No accept or reject recommendations  
- No taking sides  
- No storage of user emails or API keys  
- Output only within strict JSON schemas  

These guardrails ensure the tool supports learning and judgment rather than replacing it.

---

## Technology stack

- Frontend: Streamlit  
- LLM: OpenAI API (user-provided keys supported)  
- Architecture: Multi-step LLM chain with schema validation  
- State management: Streamlit session state  

---

## Project status

- v1 deployed and stable  
- v2 in progress with:
  - Secure API key handling  
  - Multi-step reasoning chain  
  - Coaching-oriented outputs  

Planned next steps include UI refinements, clearer evidence highlighting, and optional extensions such as job offer specific modes or multi-email thread analysis.

---

## Academic context

This project is built as part of a Negotiations and Contracts course and explicitly maps to course concepts such as:

- The negotiation checklist  
- Strategy selection  
- Power and leverage  
- Emotions in negotiation  
- Negotiation styles  

The goal is not to optimize outcomes, but to improve judgment and learning.

---

## License and use

This project is intended for educational and experimental use. It is not designed to replace professional judgment or legal advice.