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

        # next the code should extract the skills and use it to create features
        job_skills = extract_skills(processed_job_description)
        features = [[skills_match(extract_skills(resume), job_skills)] for resume in processed_resumes]

        # at this point it is due to predice the model
        predictions = model.predict(features)

        return jsonify({'rankings': predictions.tolist()})

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    app.run(debug=True)

# okay the flask i installed earlier time to put it to work in real time
# # created the flask web app to serve it, should work now
# # everything i did by 4:45am on 3/08/2024
# Data Preprocessing: Loads and cleans the data.
# Feature Engineering: Extracts and processes features.
# Model Training: Trains and saves the model.
# Model Evaluation: Evaluates model performance.
# Web Application: Handles incoming requests, integrates with the model, and provides error handling.