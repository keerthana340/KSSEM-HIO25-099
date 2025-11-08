import streamlit as st
import pdfplumber, docx, io, re, os, math, json, sqlite3
import pandas as pd
import numpy as np
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

DB_PATH = "resumes.db"

# ---------- DATABASE SETUP ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Candidate master table
    c.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        overall_score REAL,
        skills_score REAL,
        exp_score REAL,
        edu_score REAL,
        proj_score REAL,
        years INTEGER,
        num_skills INTEGER
    )
    """)
    # Chunks table
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER,
        section_name TEXT,
        section_text TEXT,
        FOREIGN KEY(candidate_id) REFERENCES candidates(id)
    )
    """)
    # Features table
    c.execute("""
    CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER,
        features_json TEXT,
        FOREIGN KEY(candidate_id) REFERENCES candidates(id)
    )
    """)
    conn.commit()
    conn.close()

def insert_candidate_data(filename, scores, feats, sections):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Insert or update candidate master
    c.execute("""
    INSERT INTO candidates(filename, overall_score, skills_score, exp_score, edu_score, proj_score, years, num_skills)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(filename) DO UPDATE SET
        overall_score=excluded.overall_score,
        skills_score=excluded.skills_score,
        exp_score=excluded.exp_score,
        edu_score=excluded.edu_score,
        proj_score=excluded.proj_score,
        years=excluded.years,
        num_skills=excluded.num_skills
    """, (
        filename, scores["overall"], scores["skills_score"], scores["experience_score"],
        scores["education_score"], scores["projects_score"], feats["years"], len(feats["skills"])
    ))
    conn.commit()
    candidate_id = c.execute("SELECT id FROM candidates WHERE filename=?", (filename,)).fetchone()[0]
    # Clear old chunks & features if re-upload
    c.execute("DELETE FROM chunks WHERE candidate_id=?", (candidate_id,))
    c.execute("DELETE FROM features WHERE candidate_id=?", (candidate_id,))
    # Insert chunks
    for sec_name, sec_text in sections.items():
        c.execute("INSERT INTO chunks(candidate_id, section_name, section_text) VALUES (?, ?, ?)",
                  (candidate_id, sec_name, sec_text))
    # Insert features JSON
    c.execute("INSERT INTO features(candidate_id, features_json) VALUES (?, ?)",
              (candidate_id, json.dumps(feats)))
    conn.commit()
    conn.close()

# ---------- TEXT EXTRACTION ----------
def extract_text_from_pdf(file_stream):
    text = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_docx(file_stream):
    doc = docx.Document(file_stream)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_bytes(name, b):
    try:
        if name.endswith(".pdf"): return extract_text_from_pdf(io.BytesIO(b))
        if name.endswith(".docx"): return extract_text_from_docx(io.BytesIO(b))
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("utf-8", errors="ignore")

# ---------- CHUNKING ----------
SECTION_HEADERS = ["education", "experience", "skills", "projects", "certifications"]
def chunk_by_sections(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sections = defaultdict(list)
    current = "other"
    for line in lines:
        low = line.lower()
        matched = [h for h in SECTION_HEADERS if h in low]
        if matched:
            current = matched[0]
        else:
            sections[current].append(line)
    return {k: "\n".join(v) for k,v in sections.items()}

# ---------- FEATURE EXTRACTION ----------
SKILL_LIST = ["python","java","react","node","sql","aws","docker","flask","tensorflow","javascript","c++"]

def extract_skills(text):
    low = text.lower()
    found = [s for s in SKILL_LIST if re.search(rf"\b{s}\b", low)]
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in SKILL_LIST and token.text.lower() not in found:
            found.append(token.text.lower())
    return sorted(set(found))

def extract_years(text):
    matches = re.findall(r"(\d+)\s+years?", text.lower())
    if matches:
        return max([int(x) for x in matches])
    return 0

def extract_education_level(text):
    levels = {"phd":5,"masters":4,"bachelor":3,"diploma":2}
    for k,v in levels.items():
        if k in text.lower():
            return v
    return 1

# ---------- SCORING ----------
def score_candidate(features, weights):
    skill_score = min(1.0, len(features["skills"]) / 10)
    exp_score = min(1.0, features["years"] / 10)
    edu_score = features["edu_level"] / 5
    proj_score = 0.3 if "projects" in features["sections"] else 0.0
    overall = (weights["skills"]*skill_score + weights["experience"]*exp_score +
               weights["education"]*edu_score + weights["projects"]*proj_score)
    return {
        "skills_score": skill_score,
        "experience_score": exp_score,
        "education_score": edu_score,
        "projects_score": proj_score,
        "overall": round(overall,4)
    }

# ---------- FEATURE 4: RECRUITER SUMMARY ----------
def generate_recruiter_summary(features, scores):
    skills = ", ".join(features["skills"]) if features["skills"] else "No notable skills"
    exp_years = features["years"]
    edu_level_map = {5: "PhD", 4: "Masters", 3: "Bachelor", 2: "Diploma", 1: "Other"}
    edu_level = edu_level_map.get(features.get("edu_level", 1), "Other")
    projects_present = "Yes" if "projects" in features["sections"] else "No"
    
    summary = (
        f"Skills: {skills}. "
        f"Experience: {exp_years} years. "
        f"Education Level: {edu_level}. "
        f"Projects section found: {projects_present}. "
        f"Overall candidate score: {scores['overall']}. "
    )
    concerns = []
    if scores["skills_score"] < 0.3:
        concerns.append("Low skill match")
    if scores["experience_score"] < 0.3:
        concerns.append("Limited experience")
    if scores["education_score"] < 0.3:
        concerns.append("Education level below preferred")
    if concerns:
        summary += "Concerns: " + ", ".join(concerns) + "."
    
    return summary

# ---------- FEATURE 9: PDF HIGHLIGHTED SKILLS VIEWER ----------
def highlight_skills_in_text(text, skills):
    def replacer(match):
        return f"<span style='color:green'>{match.group(0)}</span>"
    for skill in skills:
        pattern = re.compile(rf"\b({re.escape(skill)})\b", re.IGNORECASE)
        text = pattern.sub(replacer, text)
    return text

# ---------- FEATURE 10: AI-Powered Resume Improvement FEEDBACK ----------
def improvement_feedback(features):
    feedback = []
    if len(features["skills"]) < 3:
        feedback.append("Consider adding more relevant technical skills to your resume.")
    if features["years"] < 2:
        feedback.append("Highlight any internships or projects to demonstrate experience.")
    if features["edu_level"] < 3:
        feedback.append("Consider pursuing higher education or certifications.")
    if "projects" not in features["sections"]:
        feedback.append("Add a Projects section to showcase your work.")
    if not feedback:
        feedback.append("Your resume looks good. Keep it updated and tailored for each job.")
    return " ".join(feedback)

# ---------- UI ----------
st.set_page_config(page_title="ResumeRank with Student & HR Portals", layout="wide")

portal = st.sidebar.radio("Select Portal", ["Student Portal", "HR Portal"])

# ---------------------------------------
if portal == "Student Portal":
    st.title("ResumeRank — Student Portal: Resume Upload & Improvement")

    init_db()

    with st.sidebar:
        st.header("Weights")
        w_skills = st.slider("Skills", 0.0, 1.0, 0.4)
        w_exp = st.slider("Experience", 0.0, 1.0, 0.35)
        w_edu = st.slider("Education", 0.0, 1.0, 0.15)
        w_proj = st.slider("Projects", 0.0, 1.0, 0.1)
        weights = {"skills":w_skills,"experience":w_exp,"education":w_edu,"projects":w_proj}
        st.divider()
        uploaded = st.file_uploader("Upload resumes", type=["pdf","docx","txt"], accept_multiple_files=True)

    if uploaded:
        results = []
        for file in uploaded:
            raw = file.read()
            txt = extract_text_from_bytes(file.name, raw)
            sections = chunk_by_sections(txt)
            feats = {
                "skills": extract_skills(txt),
                "years": extract_years(txt),
                "edu_level": extract_education_level(txt),
                "sections": sections
            }
            scores = score_candidate(feats, weights)
            insert_candidate_data(file.name, scores, feats, sections)
            
            summary = generate_recruiter_summary(feats, scores)
            feedback = improvement_feedback(feats)
            
            results.append({
                "filename": file.name,
                "overall": scores["overall"],
                "skills": len(feats["skills"]),
                "years": feats["years"],
                "summary": summary,
                "feedback": feedback,
                "raw_text": txt,
                "highlighted_text": highlight_skills_in_text(txt, feats["skills"])
            })
        st.success("Data stored in database successfully!")
        
        df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ['raw_text', 'highlighted_text', 'summary', 'feedback']} for r in results])
        st.dataframe(df_res)
        
        st.divider()
        
        for res in results:
            st.subheader(f"Summary for {res['filename']}")
            st.markdown(res["summary"])
            st.subheader(f"Improvement Feedback for {res['filename']}")
            st.markdown(res["feedback"])
            st.subheader(f"Resume Preview with Skills Highlighted: {res['filename']}")
            st.markdown(res["highlighted_text"].replace("\n", "<br>"), unsafe_allow_html=True)
        
    else:
        st.info("Upload resumes to start.")

    st.divider()
    st.subheader("Database Summary")
    if st.button("Show stored candidates"):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM candidates ORDER BY overall_score DESC", conn)
        st.dataframe(df)
        conn.close()

    if st.button("Show Analytics Dashboard"):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM candidates ORDER BY overall_score DESC", conn)
        
        st.subheader("Skill Count Distribution")
        st.bar_chart(df["num_skills"])
        
        st.subheader("Years of Experience Distribution")
        st.bar_chart(df["years"])
        
        st.subheader("Education Level Distribution")
        edu_map = {5: "PhD", 4: "Masters", 3: "Bachelor", 2: "Diploma", 1: "Other"}
        edu_names = df["edu_score"].apply(lambda x: edu_map.get(round(x*5), "Other"))
        edu_counts = edu_names.value_counts()
        st.bar_chart(edu_counts)
        
        st.subheader("Candidates Overall Scores")
        st.line_chart(df["overall_score"])
        conn.close()

# ---------------------------------------
elif portal == "HR Portal":
    st.title("ResumeRank — HR Portal: Candidate Ranking by Job Description")
    st.markdown("Upload or paste Job Description texts to rank candidates based on combined skill match.")
    
    jd_text = st.text_area("Paste combined Job Description Text Here", height=200)
    jd_files = st.file_uploader("Or upload Job Description files (.txt, .pdf, .docx)", 
                                type=["txt", "pdf", "docx"], key="jdfiles", accept_multiple_files=True)
    
    # If multiple files uploaded and no paste, combine their texts
    if jd_files and not jd_text.strip():
        combined_texts = []
        for jd_file in jd_files:
            jd_raw = jd_file.read()
            combined_texts.append(extract_text_from_bytes(jd_file.name, jd_raw))
        jd_text = "\n\n".join(combined_texts)
    
    if jd_text.strip():
        jd_skills = extract_skills(jd_text)
        st.markdown(f"*Extracted JD Skills (combined):* {', '.join(jd_skills) if jd_skills else 'No skills found.'}")
        
        conn = sqlite3.connect(DB_PATH)
        df_candidates = pd.read_sql_query("SELECT * FROM candidates", conn)
        features_df = pd.read_sql_query("SELECT candidate_id, features_json FROM features", conn)
        conn.close()
        
        candidate_scores = []
        for _, row in df_candidates.iterrows():
            candidate_id = row['id']
            features_json = features_df[features_df["candidate_id"]==candidate_id]["features_json"]
            if features_json.empty:
                continue
            feats = json.loads(features_json.values[0])
            c_skills = set(feats.get("skills", []))
            c_years = feats.get("years", 0)
            
            if jd_skills and c_skills:
                intersection = c_skills.intersection(set(jd_skills))
                union = c_skills.union(set(jd_skills))
                jd_match = len(intersection) / len(union) if union else 0.0
            else:
                jd_match = 0.0
            
            candidate_scores.append({
                "filename": row["filename"],
                "overall_score": row["overall_score"],
                "jd_match_score": round(jd_match, 4),
                "years": c_years
            })
        
        if candidate_scores:
            ranked_df = pd.DataFrame(candidate_scores)
            ranked_df = ranked_df.sort_values(by=["jd_match_score", "overall_score"], ascending=False)
            st.subheader("Candidates Ranked by Combined Job Description Match")
            st.dataframe(ranked_df)
            
            if st.checkbox("Show Top Candidate Summaries", value=True):
                for _, c_row in ranked_df.head(10).iterrows():
                    filename = c_row["filename"]
                    idx = df_candidates[df_candidates["filename"]==filename].index[0]
                    conn = sqlite3.connect(DB_PATH)
                    features_json_str = pd.read_sql_query(f"SELECT features_json FROM features WHERE candidate_id={df_candidates.loc[idx,'id']}", conn).iloc[0,0]
                    conn.close()
                    feats = json.loads(features_json_str)
                    scores = {
                        "overall": df_candidates.loc[idx,"overall_score"],
                        "skills_score": df_candidates.loc[idx,"skills_score"],
                        "experience_score": df_candidates.loc[idx,"exp_score"],
                        "education_score": df_candidates.loc[idx,"edu_score"],
                        "projects_score": df_candidates.loc[idx,"proj_score"]
                    }
                    summ = generate_recruiter_summary(feats, scores)
                    st.markdown(f"{filename} Summary:** {summ}")
        else:
            st.info("No candidates stored in database yet.")
    else:
        st.info("Enter or upload Job Description text(s) to rank candidates.")