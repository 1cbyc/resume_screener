from flask import Flask, request, jsonify
import joblib
import logging
from data_preprocessing import preprocess_text
from feature_engineering import extract_skills, skills_match

app = Flask(__name__)
# model = joblib.load('model.pkl')

# let me configure logging this time
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

# should be able to load model with error handling now atleast
try:
    model = joblib.load('model.pkl')
except (EOFError, FileNotFoundError) as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

@app.route('/rank', methods=['POST'])
def rank():
    if model is None:
        return jsonify({'error': 'This Model no load well'}), 500

    try:
        # let me just extract data from request first (will come back later)
        data = request.json
        if 'resumes' not in data or 'job_description' not in data:
            return jsonify({'error': 'Invalid Input data'}), 400

        resumes = data['resumes']
        job_description = data['job_description']

    # resumes = request.json['resumes']
    # job_description = request.json['job_description']

    # time to preprocess resume and job desc
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