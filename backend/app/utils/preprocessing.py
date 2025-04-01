import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64

def preprocess_image(image_bytes):
    """Preprocess image bytes for model input"""
    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and normalize
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    
    return image_array

def generate_explanations(model, image_array):
    """Generate Grad-CAM explanations"""
    # Prepare image batch
    image_batch = np.expand_dims(image_array, 0)
    
    # Generate Grad-CAM
    grad_cam = generate_gradcam(model, image_array)
    
    # Create visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array / 255.0)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(grad_cam, cmap='jet', alpha=0.7)
    plt.title('Grad-CAM Explanation')
    
    # Convert matplotlib figure to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    explanation_png = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        'grad_cam': grad_cam.tolist(),
        'visualization': explanation_png,
    }

def generate_gradcam(model, image_array):
    """Generate Grad-CAM visualization"""
    # Get the last conv layer
    last_conv_layer = model.get_last_conv_layer()
    
    # Create Grad-CAM model
    grad_model = tf.keras.Model(
        [model.model.inputs],
        [last_conv_layer.output, model.model.output]
    )
    
    # Prepare input
    image_batch = np.expand_dims(image_array, 0)
    
    # Generate Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        class_idx = tf.argmax(predictions[0])
        output = predictions[:, class_idx]
        
    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Resize heatmap to match input image size
    heatmap = tf.image.resize(
        tf.expand_dims(tf.expand_dims(heatmap, -1), 0),
        (224, 224)
    )[0, :, :, 0]
    
    return heatmap.numpy() 