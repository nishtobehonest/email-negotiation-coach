# Email Negotiation Coach (v1)

A reflection-first negotiation coaching tool designed to help users slow down, diagnose negotiation context, and apply negotiation theory before replying to emails.

This project is intentionally human-in-the-loop. It does not draft replies or send emails. Instead, it helps users understand what is happening in a negotiation and identify preparation gaps before they respond.

---

## Why this project

In negotiation classes and real life, people often react emotionally or instinctively to emails, especially under time pressure. While frameworks like the negotiation checklist are taught, they are rarely applied in real time.

This tool bridges that gap by applying negotiation concepts live, in a structured and reflective way.

---

## What the app does (v1)

Given:
- An incoming negotiation email
- Optional user context such as stakes or relationship importance

The app provides:
- A structured diagnosis of the negotiation context
  - Negotiation type (distributive, integrative, mixed)
  - Negotiation stage (opening, bargaining, closing)
  - Power signals (deadlines, alternatives, authority)
  - Emotional tone with supporting evidence
- Identification of preparation gaps using the Negotiation Checklist (Reading 1.4)
- Brief explanations grounded in course concepts such as power, emotions, and negotiation style

The app intentionally does NOT:
- Draft or suggest email replies
- Automate sending emails
- Replace human judgment

---

## Design philosophy

This tool is designed to support learning and reflection, not automation. It acts as a coaching layer that helps users pause, diagnose, and think clearly before acting.

The emphasis is on:
- Cognitive distance
- Theory application
- Structured reasoning
- Human-in-the-loop decision making

---

## Technical overview (v1)

- Frontend: Streamlit
- Core logic: Single LLM call with structured JSON output
- Prompting: Grounded in Lewicki et al. readings
- Secrets handling: Environment variables using dotenv
- Deployment: Streamlit Cloud

---

## Status

Version 1 is a stable baseline focused on diagnosis and checklist-based reflection.

Planned future work includes:
- Secure user-provided API keys
- Chained reasoning steps
- Workflow orchestration using tools like n8n
- Optional reply drafting with explicit human approval

---

## Disclaimer

This tool is for educational purposes only. It does not provide legal or professional advice and should not be used to automate real-world negotiations without human oversight.