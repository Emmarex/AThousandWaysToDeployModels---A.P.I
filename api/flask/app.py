# 
import json
import pickle
# 
from flask import Flask, request, abort, jsonify

flask_app = Flask(__name__)

movie_model_path = "../Models/MovieRecommenderSystem/movie_classifier.pkl"

@flask_app.route('/', methods=['GET'])
def index_page():
    return_data = {
        "error" : "0",
        "message" : "Successful"
    }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

@flask_app.route('/predict', methods=['POST'])
def predict_movie_sentiment():
    try:
        movie_review = request.form.get("movie_review")
        movie_classifier = pickle.loads(open(movie_model_path, 'rb'))
        return_data = {
            "error" : "0",
            "message" : ""
        }
    except Exception as e:
        return_data = {
            "error" : "1",
            "message" : str(e)
        }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

if __name__ == "__main__":
    flask_app.run(host='127.0.0.1', port=8080, debug=True)