import streamlit as st
import pandas as pd

from resume_screening import ScreeningConfig, get_sample_inputs, screen_resumes


st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("AI Resume Screening System (MVP)")
st.caption("Upload a job description + multiple resumes to get match score, ranking, strengths, gaps, and fit recommendation.")


sample_jd, sample_resumes = get_sample_inputs()

with st.sidebar:
    st.header("Inputs")
    jd_text = st.text_area("Job Description", value=sample_jd, height=220)
    st.divider()
    st.subheader("Resumes")
    uploaded = st.file_uploader(
        "Upload .txt resumes (multiple allowed)",
        type=["txt"],
        accept_multiple_files=True,
    )

    use_samples = st.toggle("Use sample resumes", value=(uploaded is None or len(uploaded) == 0))

    top_n = st.slider("Top JD keywords", 8, 30, 15)

    run = st.button("Run Screening")


def _load_uploaded_resumes(files):
    out = {}
    for f in files:
        name = f.name.rsplit(".", 1)[0]
        raw = f.read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")
        out[name] = text
    return out


resumes = {}
if uploaded is not None and len(uploaded) > 0 and not use_samples:
    resumes = _load_uploaded_resumes(uploaded)
else:
    resumes = sample_resumes


if run:
    if not jd_text.strip():
        st.error("Please provide a job description.")
        st.stop()
    if not resumes:
        st.error("Please upload at least one resume or enable sample resumes.")
        st.stop()

    config = ScreeningConfig(top_n_keywords=top_n)
    results = screen_resumes(jd_text, resumes, config=config)

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

    st.subheader("Ranked Results")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Top Picks")
    st.write(df.head(3))

    st.subheader("Download")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="resume_screening_results.csv",
        mime="text/csv",
    )

else:
    st.info("Press `Run Screening` to generate candidate ranking using the MVP logic.")

