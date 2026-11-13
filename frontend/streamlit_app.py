"""Streamlit interface for submitting an AI response for later evaluation."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Response Validation", page_icon="🧪", layout="centered")
st.title("AI Response Validation System")
st.caption("Milestone 1 — Evaluation input submission")
st.info(
    "This milestone accepts and stores submissions. Automated judge-agent scoring "
    "will be implemented in a later milestone."
)

with st.form("evaluation_submission"):
    question = st.text_area("Question", placeholder="Enter the original question...", height=100)
    ai_response = st.text_area("AI-generated response", placeholder="Paste the AI response...", height=180)
    reference_answer = st.text_area(
        "Reference answer (optional)", placeholder="Paste a trusted answer, if available...", height=100
    )
    source_document = st.text_area(
        "Source document / reference material (optional)",
        placeholder="Paste relevant source material, if available...",
        height=140,
    )
    submitted = st.form_submit_button("Submit for evaluation", type="primary")

if submitted:
    if not question.strip() or not ai_response.strip():
        st.error("Question and AI-generated response are required.")
    else:
        payload = {
            "question": question,
            "ai_response": ai_response,
            "reference_answer": reference_answer or None,
            "source_document": source_document or None,
        }
        try:
            response = requests.post(f"{API_BASE_URL}/submissions", json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            st.success("Submission stored successfully.")
            st.write(f"Submission ID: `{result['submission_id']}`")
            st.caption(result["message"])
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_BASE_URL}. Start the FastAPI server and try again.")
            st.caption(str(exc))
