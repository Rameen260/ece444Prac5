from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)

# Define API route for prediction
@app.route('/predict/<text>', methods=['GET'])
def predict(text):
    # Model loading
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    # Vectorize the input text and make prediction
    prediction = loaded_model.predict(vectorizer.transform([text]))[0]

    # Directly return the prediction as it is
    return jsonify({'prediction': prediction})

# Start the Flask app
if __name__ == '__main__':
    # Set the port to run on 5000 or the environment's PORT variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
