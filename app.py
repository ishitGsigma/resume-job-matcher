import streamlit as st
import pickle
import re
import nltk
import PyPDF2
import docx
import io
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        return ' '.join(words)


@st.cache_resource
def load_model_files():
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    with open('cleaner.pkl', 'rb') as f:
        cln = pickle.load(f)
    return vec, cln


def read_pdf(file):
    pdf = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text


def read_docx(file):
    doc = docx.Document(io.BytesIO(file.getvalue()))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Job Description Matcher")
st.markdown("---")

vec, cln = load_model_files()

tab1, tab2 = st.tabs(["📤 Upload Resume", "📋 Job Description"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose your resume",
        type=['pdf', 'docx', 'txt'],
        help="PDF, DOCX, or TXT format"
    )

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_raw = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_raw = read_docx(uploaded_file)
        else:
            resume_raw = uploaded_file.getvalue().decode("utf-8")

        st.success("✅ Resume uploaded successfully!")
        with st.expander("Preview resume text"):
            st.text(resume_raw[:800] + "..." if len(resume_raw) > 800 else resume_raw)
    else:
        resume_raw = None

with tab2:
    job_raw = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="Copy and paste the job posting..."
    )

    if job_raw:
        st.success("✅ Job description added!")
        with st.expander("Preview job text"):
            st.text(job_raw[:800] + "..." if len(job_raw) > 800 else job_raw)

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button(
        "🔍 ANALYZE MATCH",
        type="primary",
        use_container_width=True,
        disabled=not (uploaded_file and job_raw)
    )

if analyze_btn:
    with st.spinner("Analyzing your resume..."):
        resume_clean = cln.clean(resume_raw)
        job_clean = cln.clean(job_raw)

        vectors = vec.transform([resume_clean, job_clean])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        match_score = round(similarity * 100, 1)

        st.markdown("---")

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col2:
            if match_score >= 70:
                st.markdown(f"<h1 style='text-align: center; color: #00cc66;'>{match_score}%</h1>",
                            unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: center;'><span style='background: #00cc66; color: white; padding: 5px 15px; border-radius: 20px;'>⭐ EXCELLENT MATCH</span></p>",
                    unsafe_allow_html=True)
            elif match_score >= 50:
                st.markdown(f"<h1 style='text-align: center; color: #ffaa33;'>{match_score}%</h1>",
                            unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: center;'><span style='background: #ffaa33; color: white; padding: 5px 15px; border-radius: 20px;'>📈 GOOD MATCH</span></p>",
                    unsafe_allow_html=True)
            elif match_score >= 30:
                st.markdown(f"<h1 style='text-align: center; color: #ff6633;'>{match_score}%</h1>",
                            unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: center;'><span style='background: #ff6633; color: white; padding: 5px 15px; border-radius: 20px;'>⚠️ FAIR MATCH</span></p>",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='text-align: center; color: #ff4444;'>{match_score}%</h1>",
                            unsafe_allow_html=True)
                st.markdown(
                    "<p style='text-align: center;'><span style='background: #ff4444; color: white; padding: 5px 15px; border-radius: 20px;'>❌ LOW MATCH</span></p>",
                    unsafe_allow_html=True)

        st.markdown("---")

        features = vec.get_feature_names_out()
        job_vector = vectors[1].toarray()[0]
        resume_vector = vectors[0].toarray()[0]

        job_words = []
        for i in range(len(features)):
            if job_vector[i] > 0.05:
                job_words.append((features[i], job_vector[i]))
        job_words.sort(key=lambda x: x[1], reverse=True)

        resume_words = []
        for i in range(len(features)):
            if resume_vector[i] > 0.05:
                resume_words.append((features[i], resume_vector[i]))

        resume_dict = dict(resume_words)

        missing = []
        for word, score in job_words[:25]:
            if word not in resume_dict:
                missing.append((word, round(score * 100, 1)))

        col_k1, col_k2 = st.columns(2)

        with col_k1:
            st.subheader("📌 Top Resume Keywords")
            if resume_words:
                for word, score in resume_words[:10]:
                    st.markdown(f"- **{word}**")
            else:
                st.info("No keywords detected")

        with col_k2:
            st.subheader("📌 Top Job Keywords")
            if job_words:
                for word, score in job_words[:10]:
                    bar_width = int(score * 100)
                    st.markdown(f"- **{word}**")
            else:
                st.info("No keywords detected")

        st.markdown("---")
        st.subheader("🎯 Skills to Add")

        if missing:
            cols = st.columns(3)
            for i, (skill, imp) in enumerate(missing[:12]):
                col_idx = i % 3
                with cols[col_idx]:
                    if imp > 20:
                        st.markdown(
                            f"<span style='background: #ff4444; color: white; padding: 8px; border-radius: 10px; display: block; text-align: center; margin: 5px;'><b>{skill}</b> 🔥</span>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<span style='background: #ffaa33; color: white; padding: 8px; border-radius: 10px; display: block; text-align: center; margin: 5px;'><b>{skill}</b></span>",
                            unsafe_allow_html=True)
        else:
            st.success("✨ Perfect! Your resume contains all key skills!")

        st.markdown("---")
        st.subheader("💡 Recommendations")

        if missing:
            st.markdown("Based on the job description, consider adding:")
            for skill, imp in missing[:5]:
                if imp > 15:
                    st.markdown(f"• **{skill}** - high priority (present in {imp}% of job keywords)")
                else:
                    st.markdown(f"• **{skill}** - good to have")

            if match_score < 60:
                st.info("💪 Add these missing skills to your resume and try again!")
        else:
            st.balloons()
            st.success("Your resume is well-optimized for this position!")

st.markdown("---")
st.caption("Upload your resume and paste job description to see how well you match")