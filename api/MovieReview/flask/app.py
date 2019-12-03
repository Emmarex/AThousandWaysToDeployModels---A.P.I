#
import json
import pickle
import numpy as np
from flask import Flask, request
# 
from helper import vectorizer

flask_app = Flask(__name__)

classifier_folder_path = "Model/movie_classifier.pkl"

@flask_app.route('/', methods=['GET'])
def index_page():
    return_data = {
        "error" : "0",
        "message" : "Successful"
    }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

@flask_app.route('/predict', methods=['POST'])
def classify_movie_review():
    try:
        if "movie_review" in request.form and request.form.get("movie_review") is not None:
            movie_review = request.form.get("movie_review")
            movie_review = [movie_review]
            vectorized_review = vectorizer.transform(movie_review)
            movie_classifier = pickle.load(open(classifier_folder_path, 'rb'))
            prediction = movie_classifier.predict(vectorized_review)
            prediction_prob = np.max(movie_classifier.predict_proba(vectorized_review)) * 100
            review_sentiment = "Positive" if prediction[0] == 1 else "Negative"
            return_data = {
                "error" : "0",
                "message" : "Successful",
                "review_sentiment" : review_sentiment
            }
        else:
            return_data = {
                "error" : "1",
                "message" : "Invalid parameters"
            }
    except Exception as e:
        return_data = {
            "error" : "1",
            "message" : f"[Error] : {e}"
        }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

if __name__ == "__main__":
    flask_app.run(port=8080, debug=False)