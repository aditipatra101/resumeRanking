# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64 #what does base 64 meaning?? Converts files (like Excel) into a format you can download in browser.It converts binary files (like Excel, images) into text format.This helps in downloading files directly from your Streamlit app using an HTML link.

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Datasets
job_df = pd.read_excel("job_title_des.xlsx")
resume_df = pd.read_excel("gpt_dataset.xlsx")

# Text cleaning function
def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Preprocess resumes and job descriptions
resume_df["Cleaned_Resume"] = resume_df["Resume"].apply(clean_text)
job_df["Cleaned_JD"] = job_df["Job Description"].apply(clean_text)

# Streamlit UI
st.title("AI-Powered Resume Screening Tool")
st.write("Upload a job description and get ranked resumes based on relevance.")

# Job title dropdown
job_titles = job_df["Job Title"].unique()
selected_title = st.selectbox("Select Job Title", job_titles)

# Get selected job description
job_text = job_df[job_df["Job Title"] == selected_title]["Cleaned_JD"].values[0]

# TF-IDF vectorization
corpus = resume_df["Cleaned_Resume"].tolist() + [job_text]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Cosine similarity
cos_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
resume_df["Score"] = cos_sim.flatten()
ranked_df = resume_df.sort_values(by="Score", ascending=False)

# Show ranked results
st.subheader("Ranked Resumes")
st.dataframe(ranked_df[["Category", "Score", "Resume"]].reset_index(drop=True))

# Export to Excel function
import io

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# Download button
if st.button("Download Ranked Results as Excel"):
    excel_data = convert_df_to_excel(ranked_df[["Category", "Score", "Resume"]])
    b64 = base64.b64encode(excel_data).decode()
    download_link = '<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + b64 + '" download="ranked_resumes.xlsx">Download Excel File</a>'
    st.markdown(download_link, unsafe_allow_html=True)






