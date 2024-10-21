from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load the pre-trained model
model = load_model('handwritten_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, 28, 28, 1) / 255.0

    prediction = model.predict(image)
    predicted_class = chr(np.argmax(prediction) + 65)  # Convert 0-25 to A-Z

    return jsonify(result=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
