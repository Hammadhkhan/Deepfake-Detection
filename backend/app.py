import os
# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)

# --- Model Loading ---
# Load a pre-trained Xception model from Keras applications.
# This model is trained on ImageNet and will serve as the backbone.
# For a real-world application, this should be replaced with a model
# fine-tuned on a deepfake detection dataset (e.g., FaceForensics++).
try:
    model = tf.keras.applications.Xception(weights='imagenet')
    # We need to find the last convolutional layer for Grad-CAM
    last_conv_layer_name = "block14_sepconv2_act"
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    last_conv_layer_name = None

# --- Image Preprocessing ---
def preprocess_image(image_bytes, target_size=(299, 299)):
    """
    Preprocesses an image for the Xception model.
    - Decodes image bytes
    - Resizes to target size
    - Converts to numpy array
    - Preprocesses for Xception (scales pixel values)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array

# --- Grad-CAM Implementation ---
def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """
    Overlays a heatmap onto the original image.
    """
    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + original_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Render."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint. Takes an image, runs inference, and returns
    prediction, confidence, and a Grad-CAM heatmap.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        image_bytes = file.read()
        preprocessed_img = preprocess_image(image_bytes)

        # --- Prediction ---
        preds = model.predict(preprocessed_img)
        # For this example, we'll treat the top prediction from ImageNet as our "confidence".
        # A real deepfake model would output a binary classification (REAL/FAKE).
        # We'll arbitrarily label top classes as FAKE and others as REAL for demonstration.
        decoded_preds = tf.keras.applications.xception.decode_predictions(preds, top=1)[0]
        top_pred = decoded_preds[0]
        confidence = float(top_pred[2])
        # This is a placeholder logic. A real model would be binary.
        prediction_label = "FAKE" if confidence > 0.5 else "REAL"

        # --- Grad-CAM Heatmap ---
        # Generate heatmap
        heatmap = get_gradcam_heatmap(preprocessed_img, model, last_conv_layer_name)

        # Overlay heatmap on original image
        original_img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_img_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
        superimposed_img = overlay_heatmap(original_img_cv, heatmap)

        # Encode superimposed image to base64
        _, img_encoded = cv2.imencode('.jpg', superimposed_img)
        heatmap_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence,
            'heatmap': f"data:image/jpeg;base64,{heatmap_base64}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
