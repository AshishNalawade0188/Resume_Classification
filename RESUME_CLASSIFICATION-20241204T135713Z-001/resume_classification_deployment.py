#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
import pandas as pd 
import numpy as np 


# In[5]:


model = pickle.load(open("vector.pkl", 'rb'))


# In[6]:


model2 = pickle.load(open("modelDT.pkl", 'rb'))


# In[7]:


# Import Libraries
import re
import docx2txt
import pdfplumber
import pandas as pd
import streamlit as st
import pickle as pk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Initialize NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load your model and vectorizer
model = pk.load(open('modelDT.pkl', 'rb'))
Vectorizer = pk.load(open('vector.pkl', 'rb'))

# Title of the application
st.title('Resume Classification')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.subheader('Upload your resumes in PDF or DOCX format')

# Function to extract text from uploaded files
def getText(doc_file):
    fullText = ''
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        fullText = docx2txt.process(doc_file)
    elif doc_file.type == "application/pdf":
        with pdfplumber.open(doc_file) as pdf:
            for page in pdf.pages:
                page_content = page.extract_text()
                if page_content:
                    fullText += page_content
    return fullText

# Preprocessing function
def preprocess(text):
    sentence = str(text)
    sentence = sentence.lower()
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 and w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

# File uploader
upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

# Initialize data structures
filename = []
predicted = []
skills = []

if upload_file:
    for doc_file in upload_file:
        if doc_file is not None:
            # Extract text from the uploaded file
            extText = getText(doc_file)
            cleaned_text = preprocess(extText)

            # Predict the profile
            # Assuming you have a function to make predictions
            # For demonstration, we will use a simple model prediction
            # You may need to transform the cleaned text using your Vectorizer
            prediction = model.predict(Vectorizer.transform([cleaned_text]))
            predicted.append(prediction[0])  # Assuming the prediction returns a list
            filename.append(doc_file.name)
            skills.append("Extracted Skills")  # Replace with actual skill extraction logic

    # Create a DataFrame to display the results
    file_type = pd.DataFrame({
        'Uploaded File': filename,
        'Skills': skills,
        'Predicted Profile': predicted
    })

    # Display the results in a table
    if len(predicted) > 0:
        st.table(file_type.style.format())

    # Selection for filtering profiles
    select = ['PeopleSoft', 'SQL Developer', 'React JS Developer', 'Workday']
    st.subheader('Select as per Requirement')
    option = st.selectbox('Fields', select)

    # Filter and display results based on selection
    if option in file_type['Predicted Profile'].values:
        st.table(file_type[file_type['Predicted Profile'] == option])
    else:
        st.write("No resumes found for the selected profile.")


# In[ ]:




