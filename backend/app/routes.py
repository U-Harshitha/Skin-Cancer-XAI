from flask import Blueprint, request, jsonify
from .models.cnn_model import SkinCancerModel
from .utils.preprocessing import preprocess_image, generate_explanations
import tensorflow as tf
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)
model = SkinCancerModel()

@api.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        logger.debug(f"Received image file: {image_file.filename}")
        
        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        image_bytes = image_file.read()
        logger.debug("Image bytes read successfully")
        
        # Preprocess image
        try:
            processed_image = preprocess_image(image_bytes)
            logger.debug("Image preprocessed successfully")
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return jsonify({'error': f'Error preprocessing image: {str(e)}'}), 500
        
        # Get prediction and explanations
        try:
            prediction_result = model.predict(processed_image)
            logger.debug("Prediction generated successfully")
            explanations = generate_explanations(model, processed_image)
            logger.debug("Explanations generated successfully")
        except Exception as e:
            logger.error(f"Error in prediction/explanation: {str(e)}")
            return jsonify({'error': f'Error generating prediction: {str(e)}'}), 500
        
        # Combine results
        response = {
            'predictions': prediction_result['predictions'],
            'explanations': {
                'grad_cam': explanations['grad_cam'],
                'visualization': explanations['visualization']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500 