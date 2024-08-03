def extract_skills(text):
    skills = ['python', 'java', 'machine learning', 'nlp']
    extracted_skills = [skill for skill in skills if skill in text.lower()]
    return extracted_skills

def skills_match(resume_skills, job_skills):
    return len(set(resume_skills) & set(job_skills)) / len(set(job_skills))

def create_features(data):
    data['resume_skills'] = data['processed_resumes'].apply(extract_skills)
    data['job_skills'] = data['processed_job_descriptions'].apply(extract_skills)
    data['skills_match'] = data.apply(lambda x: skills_match(x['resume_skills'], x['job_skills']), axis=1)
    return data[['skills_match']], data['label']

# if nobody will i think making the features from the preprocessed data is the best thing to do