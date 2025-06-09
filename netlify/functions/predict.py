import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from PIL import Image
import io

# === Path to the model ===
model_path = '/tmp/saved_model_palm_disease.keras'

# Download the model if it doesn't exist in the temporary directory
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id=1g-QPUIsySVm1oBl0KXpKKlxe7x_JPe7B'
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Class labels
labels = ['Boron Excess', 'Ganoderma', 'Healthy', 'Scale insect']

def handler(event, context):
    if event['httpMethod'] != 'POST':
        return {
            'statusCode': 405,
            'body': 'Method Not Allowed'
        }

    try:
        # Decode the base64 encoded image from the request body
        body = base64.b64decode(event['body'])
        
        # Open the image and resize it
        img = Image.open(io.BytesIO(body)).convert("RGB").resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return {
            'statusCode': 200,
            'body': json.dumps({
                'class': labels[class_index],
                'confidence': confidence
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }