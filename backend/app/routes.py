from flask import Blueprint, request, jsonify
from .models.cnn_model import SkinCancerModel
from .utils.preprocessing import preprocess_image, generate_explanations
import tensorflow as tf
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask Blueprint
api = Blueprint('api', __name__)

# Load model once on startup
model = SkinCancerModel()

@api.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logger.warning("No image part in the request")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    logger.debug(f"Image file received: {image_file.filename}")

    if image_file.filename == '':
        logger.warning("Empty filename submitted")
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = image_file.read()
        logger.debug("Image bytes read")

        # Check if it's a valid image format
        if not imghdr.what(None, h=image_bytes):
            logger.warning("Invalid image format")
            return jsonify({'error': 'Invalid image format'}), 400

        # Preprocess image
        try:
            processed_image = preprocess_image(image_bytes)
            logger.debug("Image preprocessing completed")
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return jsonify({'error': f'Error preprocessing image: {str(e)}'}), 500

        # Predict and generate explanations (Grad-CAM + SHAP)
        try:
            prediction_result = model.predict(processed_image)
            logger.debug("Prediction completed")

            explanations = generate_explanations(model, processed_image)
            logger.debug("Explanations (Grad-CAM & SHAP) generated")
        except Exception as e:
            logger.error(f"Prediction or explanation error: {e}")
            return jsonify({'error': f'Prediction/Explanation error: {str(e)}'}), 500

        # Success response
        response = {
            'predictions': prediction_result['predictions'],
            'explanations': {
                'grad_cam': explanations.get('grad_cam'),
                'visualization': explanations.get('visualization'),
                'saliency': explanations.get('saliency')
                 # SHAP visual (base64)
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Unexpected server error: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500
