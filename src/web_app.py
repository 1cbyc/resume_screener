from flask import Flask, request, jsonify
import joblib
from data_preprocessing import preprocess_text
from feature_engineering import extract_skills, skills_match

app = Flask(__name__)
model = joblib.load('model.pkl')


@app.route('/rank', methods=['POST'])
def rank():
    resumes = request.json['resumes']
    job_description = request.json['job_description']

    processed_resumes = [preprocess_text(resume) for resume in resumes]
    processed_job_description = preprocess_text(job_description)

    job_skills = extract_skills(processed_job_description)
    features = [[skills_match(extract_skills(resume), job_skills)] for resume in processed_resumes]

    predictions = model.predict(features)
    return jsonify({'rankings': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)

# okay the flask i installed earlier time to put it to work in real time
# created the flask web app to serve it, should work now