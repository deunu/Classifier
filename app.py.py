from flask import Flask, request, jsonify
import tensorflow as tf
import os

app = Flask(__name__)

# Ensure the path to the model is correct
model_path = os.path.join(os.path.dirname(__file__), 'model.keras')

# Load your trained model
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the input data as required by your model
    predictions = model.predict(data)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)

