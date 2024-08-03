import requests

url = 'http://localhost:5000/rank'
sample_data = {
    'resumes': [
        'Sample resume text from resume1.txt',
        'Sample resume text from resume2.txt'
    ],
    'job_description': 'Sample job description text from job_description1.txt'
}

response = requests.post(url, json=sample_data)
print(response.json())
