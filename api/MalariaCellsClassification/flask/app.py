#
import json
import pickle
import numpy as np
# 
from keras.models import load_model
# OR
# from tensorflow.keras.models import load_model
# 
from flask import Flask, request, abort, jsonify
# 
import helper

flask_app = Flask(__name__)

model_path = "Model/malaria_model.h5"

@flask_app.route('/', methods=['GET'])
def index_page():
    return_data = {
        "error" : "0",
        "message" : "Successful"
    }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

@flask_app.route('/classify', methods=['POST'])
def classify_malaria_cells():
    try:
        if "malaria_cell_image" in request.files and request.files['malaria_cell_image'] is not None:
            malaria_image = request.files['malaria_cell_image']
            is_successful, preprocessed_image = helper.preprocess_img(malaria_image)
            if is_successful:
                # load malaria model
                malaria_model = load_model(model_path)
                #
                score = malaria_model.predict(preprocessed_image)
                label_indx = np.argmax(score)
                classification = "Normal" if label_indx == 0 else "Infected"
                confidence_level = round(np.max(score), 2)
                return_data = {
                    "error" : "0",
                    "message" : "Successful",
                    "classification" : classification,
                    "confidence_level" : f"{str(confidence_level)}%"
                }
            else:
                return_data = {
                    "error" : "1",
                    "message" : "Image preprocessing error"
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
    flask_app.run(host='127.0.0.1', port=8080, debug=False, threaded=False)