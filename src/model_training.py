from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd
from feature_engineering import create_features
from data_preprocessing import preprocess_data, load_data

def retrain_and_save_model():
    data = load_data('data/resumes_and_jobs.csv')
    processed_data = preprocess_data(data)
    X, y = create_features(processed_data)
    

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics

# okay time to train the system to rank candidates (i hope someone helps me here over time)
# i want to retrain the model, i made a mistake:
# from sklearn.linear_model import LogisticRegression
# import joblib
# from feature_engineering import create_features
#
# def retrain_and_save_model():
#     data = pd.read_csv('resumes_and_jobs.csv')
#     processed_data = preprocess_data(data)
#     X, y = create_features(processed_data)
#     model = LogisticRegression()
#     model.fit(X, y)
#     joblib.dump(model, 'model.pkl')
#
# if __name__ == "__main__":
#     retrain_and_save_model()
