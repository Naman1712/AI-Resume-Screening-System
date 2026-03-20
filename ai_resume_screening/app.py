import streamlit as st
import pandas as pd
from io import BytesIO
from pypdf import PdfReader

from resume_screening import ScreeningConfig, get_sample_inputs, screen_resumes


st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("AI Resume Screening System")
st.caption("Upload a job description + multiple resumes to get match score, ranking, strengths, gaps, and fit recommendation.")


sample_jd, sample_resumes = get_sample_inputs()

with st.sidebar:
    st.header("Inputs")
    jd_text = st.text_area("Job Description", value=sample_jd, height=220)
    st.divider()
    st.subheader("Resumes")
    resume_source = st.radio(
        "Resume source",
        options=["Sample resumes", "Upload resumes"],
        horizontal=True,
    )

    uploaded = None
    if resume_source == "Upload resumes":
        uploaded = st.file_uploader(
            "Upload resumes (.txt or .pdf, multiple allowed)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )

    top_n = st.slider("Top JD keywords", 8, 30, 15)

    run = st.button("Run Screening")


def _load_uploaded_resumes(files):
    out = {}
    for f in files:
        name = f.name.rsplit(".", 1)[0]
        extension = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""

        if extension == "pdf":
            try:
                pdf_bytes = f.read()
                reader = PdfReader(BytesIO(pdf_bytes))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                text = "\n".join(pages).strip()
            except Exception:
                text = ""
        else:
            raw = f.read()
            try:
                text = raw.decode("utf-8")
            except Exception:
                text = raw.decode("latin-1", errors="ignore")

        if not text.strip():
            text = f"No extractable text found in {f.name}."
        out[name] = text
    return out


if "results" not in st.session_state:
    st.session_state["results"] = None

resumes = {}
if resume_source == "Sample resumes":
    resumes = sample_resumes
elif uploaded is not None and len(uploaded) > 0:
    resumes = _load_uploaded_resumes(uploaded)

if run:
    if not jd_text.strip():
        st.error("Please provide a job description.")
        st.stop()
    if not resumes:
        st.error("Please provide at least one resume to screen.")
        st.stop()

    config = ScreeningConfig(top_n_keywords=top_n)
    results = screen_resumes(jd_text, resumes, config=config)
    st.session_state["results"] = results

results = st.session_state.get("results")
if not results:
    st.info("Press `Run Screening` to generate candidate ranking using the MVP logic.")
    st.stop()

df = pd.DataFrame(
    [
        {
            "Candidate": r["candidate"],
            "Score (0-100)": r["score"],
            "Recommendation": r["recommendation"],
            "Strengths": ", ".join(r["strengths"]),
            "Gaps": ", ".join(r["gaps"]),
        }
        for r in results
    ]
)

st.subheader("Ranked Candidates")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Candidate Details")
candidate_names = [r["candidate"] for r in results]
selected = st.selectbox("Select a candidate", candidate_names, index=0)
match = next(r for r in results if r["candidate"] == selected)

col_a, col_b = st.columns(2)
col_a.metric("Match Score", f"{match['score']}/100")
col_b.metric("Recommendation", match["recommendation"])

col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Strengths**")
    st.markdown(
        "\n".join([f"- {s}" for s in match["strengths"]]) if match["strengths"] else "-"
    )
with col_d:
    st.markdown("**Key Gaps**")
    st.markdown(
        "\n".join([f"- {g}" for g in match["gaps"]]) if match["gaps"] else "-"
    )

with st.expander("How this MVP scores (quick overview)"):
    st.write(
        "Extracts a skills checklist from the JD, scores each resume using JD–resume TF-IDF similarity (normalized within the run), "
        "and blends it with keyword coverage. Strengths are the highest-weight matched skills; gaps are the most important missing skills."
    )

st.divider()
st.subheader("Top Picks (Top 3)")
st.write(df.head(3))

st.subheader("Download")
st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="resume_screening_results.csv",
    mime="text/csv",
)

