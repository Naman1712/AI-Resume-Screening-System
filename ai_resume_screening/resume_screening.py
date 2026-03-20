from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ScreeningConfig:
    top_n_keywords: int = 15
    max_strengths: int = 3
    max_gaps: int = 3

    # Weighted score: similarity (semantic-ish) vs keyword coverage (explicit skill match)
    w_similarity: float = 0.7
    w_coverage: float = 0.3


def _normalize(text: str) -> str:
    text = text.lower()
    # Keep + and # because "C#" and "C++" could matter for resumes.
    text = re.sub(r"[^a-z0-9+\s#-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sample_inputs() -> Tuple[str, Dict[str, str]]:
    """
    Sample data so the system can demonstrate working outputs immediately.
    Replace with your own JD + resumes when you run the app.
    """

    jd = (
        "We are looking for a Data Analyst to join our analytics team. "
        "You will work with cross-functional stakeholders to deliver insights and dashboards. "
        "Required skills: SQL (joins, window functions), Python (pandas, data cleaning), "
        "data analytics, Excel, and strong communication. "
        "Nice to have: Machine Learning basics, A/B testing, and dashboarding using Power BI or Tableau."
    )

    resumes: Dict[str, str] = {
        "Candidate A": (
            "Data Analyst with 3 years of experience. Skilled in SQL for analytics, "
            "including joins, CTEs, and window functions. Proficient in Python with pandas "
            "for data cleaning and feature engineering. Built KPI dashboards and reports in Power BI. "
            "Strengths include stakeholder communication and translating requirements into metrics."
        ),
        "Candidate B": (
            "BI Analyst focused on dashboarding. Expert in Power BI, Excel modeling, and reporting. "
            "Strong at visualizing trends and building interactive dashboards. "
            "Some experience with SQL but limited exposure to advanced window functions. "
            "Less experience with Python; mostly manual data prep."
        ),
        "Candidate C": (
            "Data Scientist intern working on ML projects. Strong in Python for data preprocessing "
            "and basic machine learning. Comfortable with data cleaning and experimentation. "
            "SQL experience is present but limited; mostly used simple queries. "
            "Dashboarding experience is minimal; communication is okay but not stakeholder-heavy."
        ),
        "Candidate D": (
            "Operations analyst with heavy Excel work. Created spreadsheets for reporting, "
            "track KPIs, and improved data accuracy via templates. "
            "SQL knowledge is limited to SELECT statements. "
            "No Python experience. Worked with stakeholders on weekly updates."
        ),
        "Candidate E": (
            "Entry-level candidate. Completed coursework in analytics. "
            "Basic SQL and Python fundamentals, but few real projects. "
            "Unclear experience with dashboards or stakeholder communication."
        ),
        "Candidate F": (
            "Analytics Engineer. Strong SQL (joins, CTEs, window functions) and deep Python (pandas, ETL). "
            "Built automated pipelines and documented analytics definitions. "
            "Experience with A/B testing and communicating results to business teams. "
            "Excel knowledge is moderate; dashboarding via Tableau."
        ),
    }
    return jd, resumes


def _extract_required_keywords(
    jd_text: str, top_n: int
) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract key terms from the JD to use as a 'skills checklist'.
    We use TF-IDF over the JD only (single-document TF-IDF => term importance by TF * idf).
    """
    jd_text_norm = _normalize(jd_text)

    # Prefer skill-like keyword extraction so strengths/gaps feel meaningful.
    # (This is a heuristic MVP, not a perfect NLP pipeline.)
    skill_patterns: List[Tuple[str, str]] = [
        (r"\bsql\b", "sql"),
        (r"\bpython\b", "python"),
        (r"\bpandas\b", "pandas"),
        (r"\bexcel\b", "excel"),
        (r"\bpower\s*bi\b", "power bi"),
        (r"\btableau\b", "tableau"),
        (r"\bdashboard(s)?\b", "dashboard"),
        (r"\bdashboarding\b", "dashboard"),
        (r"\banalytics\b", "analytics"),
        (r"\bdata\s*analytics\b", "data analytics"),
        (r"\bdata\s*cleaning\b", "data cleaning"),
        (r"\breporting\b", "reporting"),
        (r"\bstakeholder(s)?\b", "stakeholder"),
        (r"\bcommunication\b", "communication"),
        (r"\bmachine\s*learning\b", "machine learning"),
        (r"\bwindow\s*function(s)?\b", "window functions"),
        (r"\bjoin(s)?\b", "joins"),
        (r"\bcte(s)?\b", "cte"),
        (r"\betl\b", "etl"),
        (r"\bfeature\s*engineering\b", "feature engineering"),
        # Normalization turns "A/B testing" into "a b testing"
        (r"\ba\s*b\s*testing\b", "ab testing"),
    ]

    detected: List[str] = []
    for pattern, term in skill_patterns:
        if re.search(pattern, jd_text_norm, flags=re.IGNORECASE):
            detected.append(term)
    # Keep order while removing duplicates.
    seen = set()
    detected_unique: List[str] = []
    for t in detected:
        if t not in seen:
            detected_unique.append(t)
            seen.add(t)

    # Also compute TF-IDF term scores for fallback and gap ordering.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1)
    tfidf = vectorizer.fit_transform([jd_text_norm])
    terms = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    term_scores = {t: float(s) for t, s in zip(terms, scores)}

    # If we didn't detect enough skills, backfill with JD TF-IDF terms,
    # skipping overly generic / noisy terms.
    skip_terms = {
        "data",
        "deliver",
        "basics",
        "analyst",
        "cross",
        "team",
        "work",
        "required",
        "skills",
        "ability",
        "responsibilities",
        "experience",
    }

    required: List[str] = []
    required.extend(detected_unique[:top_n])

    if len(required) < top_n:
        ranked = sorted(term_scores.items(), key=lambda kv: kv[1], reverse=True)
        for term, _score in ranked:
            if len(required) >= top_n:
                break
            if term in skip_terms:
                continue
            if len(term) < 3:
                continue
            if term.isdigit():
                continue
            # Avoid picking words that are clearly not skills.
            if term in {"analyst", "candidate", "junior", "senior", "intern"}:
                continue
            if term not in required:
                required.append(term)

    return required, term_scores


def _term_in_text(term: str, text_norm: str) -> bool:
    # Use simple word-boundary-ish matching. Works best for single-word terms.
    # For multi-word terms we fall back to substring matching.
    term_norm = _normalize(term)
    if " " in term_norm:
        return term_norm in text_norm
    return re.search(rf"\b{re.escape(term_norm)}\b", text_norm) is not None


def screen_resumes(
    jd_text: str,
    resumes: Dict[str, str],
    config: ScreeningConfig | None = None,
) -> List[Dict[str, object]]:
    """
    Returns structured screening results for each resume.
    Output fields: candidate, score (0-100), strengths (list[str]), gaps (list[str]), recommendation (str)
    """
    if config is None:
        config = ScreeningConfig()

    required_keywords, jd_term_scores = _extract_required_keywords(
        jd_text, top_n=config.top_n_keywords
    )

    jd_norm = _normalize(jd_text)
    resume_names = list(resumes.keys())
    resume_texts = [_normalize(resumes[n]) for n in resume_names]

    corpus = [jd_norm] + resume_texts
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(corpus)

    jd_vec = tfidf[0:1]
    resume_vecs = tfidf[1:]

    # Semantic similarity proxy: cosine similarity between TF-IDF vectors.
    # Cosine values are often small and not directly comparable across datasets,
    # so we normalize within the current run.
    sims = cosine_similarity(jd_vec, resume_vecs).flatten()
    sim_min = float(np.min(sims))
    sim_max = float(np.max(sims))
    if sim_max - sim_min < 1e-9:
        sim_norms = np.full_like(sims, 0.5)
    else:
        sim_norms = (sims - sim_min) / (sim_max - sim_min)

    # Convert resume TF-IDF into term weights.
    vocab = vectorizer.get_feature_names_out()
    term_to_idx = {t: i for i, t in enumerate(vocab)}

    results: List[Dict[str, object]] = []
    for name, resume_norm, resume_row, sim_norm in zip(
        resume_names, resume_texts, resume_vecs, sim_norms
    ):
        # Keyword coverage: count how many required keywords appear.
        matched_terms: List[str] = []
        for kw in required_keywords:
            if _term_in_text(kw, resume_norm):
                matched_terms.append(kw)

        coverage = (len(matched_terms) / max(1, len(required_keywords)))

        # Choose strengths based on TF-IDF weight among required keywords that appear.
        strengths_with_weights: List[Tuple[str, float]] = []
        for kw in matched_terms:
            if kw in term_to_idx:
                strengths_with_weights.append((kw, float(resume_row[0, term_to_idx[kw]])))
            else:
                strengths_with_weights.append((kw, 0.0))

        strengths_with_weights.sort(key=lambda x: x[1], reverse=True)
        strengths = [t for t, _w in strengths_with_weights[: config.max_strengths]]

        # Choose gaps among required keywords that did NOT appear.
        missing = [kw for kw in required_keywords if kw not in matched_terms]
        missing.sort(key=lambda kw: jd_term_scores.get(kw, 0.0), reverse=True)
        gaps = missing[: config.max_gaps]

        # Final score: normalized similarity dominates, with coverage as a stabilizer.
        sim_n = float(min(max(sim_norm, 0.0), 1.0))
        raw = config.w_similarity * sim_n + config.w_coverage * coverage
        score = int(round(100 * raw))
        score = max(0, min(100, score))

        if score >= 80:
            rec = "Strong Fit"
        elif score >= 55:
            rec = "Moderate Fit"
        else:
            rec = "Not Fit"

        results.append(
            {
                "candidate": name,
                "score": score,
                "strengths": strengths,
                "gaps": gaps,
                "recommendation": rec,
                "similarity": sim_n,
                "coverage": coverage,
            }
        )

    # Rank by score desc; tie-breaker by similarity desc.
    results.sort(key=lambda r: (r["score"], r.get("similarity", 0.0)), reverse=True)
    # Remove helper fields from final output (keep ranking clean for UI).
    for r in results:
        r.pop("similarity", None)
        r.pop("coverage", None)

    return results

