import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def preprocess_data(data):
    data['processed_resumes'] = data['resume'].apply(preprocess_text)
    data['processed_job_descriptions'] = data['job_description'].apply(preprocess_text)
    return data

# what i am doing here is creating functions for loading and preprocessing data