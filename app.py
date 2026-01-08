# -*- coding: utf-8 -*-


import streamlit as st
import joblib
import re
import numpy as np
import os
from scipy.sparse import hstack

# -------------------------------------------------
# Resolve base directory (works on any computer)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# Load saved models & vectorizer (LOCAL)
# -------------------------------------------------
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
clf = joblib.load(os.path.join(BASE_DIR, "models", "difficulty_score_classifie.pkl"))
reg = joblib.load(os.path.join(BASE_DIR, "models", "difficulty_score_regressor.pkl"))

# -------------------------------------------------
# Text cleaning
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s+\-*/=<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------------------------
# Feature engineering
# -------------------------------------------------
keywords = [
    'dp', 'dynamic programming', 'graph', 'tree',
    'dfs', 'bfs', 'shortest path',
    'segment tree', 'fenwick',
    'greedy', 'binary search',
    'recursion', 'backtracking'
]

def keyword_count(text):
    return sum(text.count(k) for k in keywords)

def math_symbol_count(text):
    return len(re.findall(r'[+\-*/=<>]', text))

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title(" AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

problem_desc = st.text_area("Problem Description", height=200)
input_desc = st.text_area("Input Description", height=150)
output_desc = st.text_area("Output Description", height=150)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Difficulty"):
    if problem_desc.strip() == "":
        st.warning("Please enter the problem description.")
    else:
        combined_text = problem_desc + " " + input_desc + " " + output_desc
        cleaned = clean_text(combined_text)

        X_tfidf = tfidf.transform([cleaned])

        extra_features = np.array([[
            len(cleaned),
            math_symbol_count(cleaned),
            keyword_count(cleaned)
        ]])

        X_final = hstack([X_tfidf, extra_features])

        pred_class = clf.predict(X_final)[0]
        pred_score = reg.predict(X_final)[0]

        st.success("Prediction Successful ")
        st.markdown(f"###  Predicted Difficulty Class: **{pred_class}**")
        st.markdown(f"###  Predicted Difficulty Score: **{pred_score:.2f}**")

