from flask import Blueprint, request, jsonify
from .models.cnn_model import SkinCancerModel
from .utils.preprocessing import preprocess_image, generate_gradcam
import tensorflow as tf

api = Blueprint('api', __name__)
model = SkinCancerModel()

@api.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Get prediction
        predictions = model.predict(processed_image)
        
        # Generate explanation (Grad-CAM)
        heatmap = generate_gradcam(model.model, processed_image)
        
        return jsonify({
            'predictions': predictions,
            'heatmap': heatmap.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500 