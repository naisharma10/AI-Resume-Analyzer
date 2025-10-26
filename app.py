import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re

# Load model once at startup
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="centered")
st.title("🤖 AI Resume Analyzer")
st.markdown("Compare your resume to a job description and get a match score!")

# Text inputs
resume_text = st.text_area("📄 Paste your Resume Text", height=250)
job_text = st.text_area("💼 Paste Job Description", height=250)

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    common_words = {'and','the','for','with','that','this','from','your','have','are'}
    keywords = [w for w in words if w not in common_words]
    return set(keywords)

if st.button("🔍 Analyze"):
    if resume_text.strip() and job_text.strip():
        # Calculate semantic similarity
        embeddings = model.encode([resume_text, job_text])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        score = round(similarity * 100, 2)
        st.success(f"✅ **Match Score:** {score}%")

        # Keyword overlap
        resume_kw = extract_keywords(resume_text)
        job_kw = extract_keywords(job_text)
        overlap = resume_kw.intersection(job_kw)
        st.markdown(f"### 🔑 Common Skills & Keywords Found ({len(overlap)})")
        st.write(", ".join(sorted(overlap)) if overlap else "_No significant overlap found._")
    else:
        st.warning("Please paste both resume and job description text.")

