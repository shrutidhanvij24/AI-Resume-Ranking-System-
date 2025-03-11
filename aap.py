import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title of the app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# Upload resume dataset
st.header("Upload Resume Dataset")
uploaded_file = st.file_uploader("Upload a CSV file containing resumes", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read uploaded CSV file
    st.write("### Preview of Uploaded Resumes")
    st.write(df.head())  # Show first few rows
    
    if "Resume Text" in df.columns:
        resumes = df["Resume Text"].tolist()  # Convert resume column to a list
        
        # Function to rank resumes
        def rank_resumes(job_description, resumes):
            documents = [job_description] + resumes
            vectorizer = TfidfVectorizer().fit_transform(documents)
            vectors = vectorizer.toarray()
            job_description_vector = vectors[0]
            resume_vectors = vectors[1:]
            similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
            return similarities

        # Check if job description is provided
        if job_description:
            st.write("### Ranking Resumes...")
            similarities = rank_resumes(job_description, resumes)
            df["Similarity Score"] = similarities
            df = df.sort_values(by="Similarity Score", ascending=False)
            
            # Show ranked resumes
            st.write("### Ranked Resumes")
            st.write(df[["Resume Text", "Similarity Score"]])  # Show top-ranked resumes
