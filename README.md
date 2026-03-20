# AI Resume Screening System

This is a simple, working MVP for the Nutrabay AI Automation Intern assessment (Problem 1).

It takes:
- A **Job Description (JD)**
- Multiple **resumes** (plain `.txt`)

And outputs:
- **Match score (0-100)**
- **Candidate ranking**
- **Key strengths (2-3)**
- **Key gaps (2-3)**
- **Final recommendation**: `Strong Fit / Moderate Fit / Not Fit`

## How to run locally

1. Install dependencies:
   - `pip install -r requirements.txt` (root requirements)
2. Run the app:
   - `streamlit run ai_resume_screening/app.py`
3. Open the shown localhost URL in your browser.

## How it works (high level)

The MVP uses:
- TF-IDF keyword extraction from the JD to build a “required skills” list
- TF-IDF similarity between JD and each resume
- Keyword coverage to compute the final 0-100 score

Strengths and gaps are derived from which JD keywords appear in each resume.

## What to submit

For the assessment form, link to:
- Your code (GitHub/Drive/etc.)
- A working demo (Streamlit local link or a deployed link)
- A short approach explanation (200–300 words)

