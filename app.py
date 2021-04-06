from nlp import preprocess
from nlp import Similarity
import streamlit as st 
import numpy as np

st.title('Resume Match')

job = st.text_area('Paste your Job Description')
resume = st.text_area('Paste your Resume')
# job = st.file_uploader('Upload Job Descripltion')

job = preprocess(job)
resume = preprocess(resume)

matchPercentage = np.round((Similarity(job, resume)*100),2)

st.write("Your Resume matched", matchPercentage, '% with the job description')

