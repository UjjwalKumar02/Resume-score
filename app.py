import streamlit as st
import joblib
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity


# Load models
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/vectorizer.pkl")


# Cleaning func
def clean_text(text):
  text = text.lower()
  text = re.sub(r"[^a-z0-9\s]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


# Predict func

def predict_match(job_desc, resume_text):
  # Clean inputs
  job_desc = clean_text(job_desc)
  resume_text = clean_text(resume_text)

  # TF-IDF vectors
  job_vec = tfidf.transform([job_desc])
  resume_vec = tfidf.transform([resume_text])

  # Cosine similarity
  cos_sim = cosine_similarity(job_vec, resume_vec)[0][0]

  # Jaccard similarity
  def jaccard_similarity(a, b):
    a_set, b_set = set(a.split()), set(b.split())
    if len(a_set | b_set) == 0:
      return 0
    return len(a_set & b_set) / len(a_set | b_set)

  jacc_sim = jaccard_similarity(resume_text, job_desc)

  # Length ratio
  len_ratio = len(resume_text.split()) / (len(job_desc.split()) + 1)

  features = [[cos_sim, jacc_sim, len_ratio]]

  pred = model.predict(features)[0]
  return max(0, min(100, pred))



# Streamlit
st.set_page_config(page_title="Resume Matcher", layout="centered")

st.title("Resume Analyzer using ML")

st.write("Paste a job description and a resume below. The model will predict a **match score (0–100)**.")

# Input fields
job_desc = st.text_area("Job Description", height=150, placeholder="Enter job description here...")
resume_text = st.text_area("Resume Text", height=150, placeholder="Paste resume text here...")

if st.button("Predict Match"):
  if job_desc.strip() and resume_text.strip():
    score = predict_match(job_desc, resume_text)
    st.success(f"Predicted Match Score: **{score:.2f} / 100**")

    # Display a progress bar for visualization
    st.progress(int(score))

  else:
    st.warning("Please enter both job description and resume.")