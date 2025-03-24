import tensorflow as tf
import numpy as np
from PIL import Image
import io

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
    image_array = image_array / 255.0
    
    return image_array

def generate_gradcam(model, image_array, layer_name='conv2d_2'):
    """Generate Grad-CAM visualization for model explanation"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image_array]))
        class_idx = tf.argmax(predictions[0])
        output = predictions[:, class_idx]
        
    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy() 