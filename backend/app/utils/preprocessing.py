import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64
import shap
from tensorflow.keras.models import Model

def preprocess_image(image_bytes):
    """Convert uploaded image bytes into a normalized 224x224 RGB numpy array"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    return image_array

def generate_explanations(model, image_array):
    image_batch = np.expand_dims(image_array, 0)

    grad_cam = generate_gradcam(model, image_array)
    saliency_map = generate_saliency_map(model, image_array)
    

    # Grad-CAM visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array / 255.0)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(grad_cam, cmap='jet', alpha=0.7)
    plt.title('Grad-CAM')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    gradcam_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return {
        'grad_cam': grad_cam.tolist(),
        'visualization': gradcam_base64,
        'saliency': saliency_map
        
    }


def generate_gradcam(model_wrapper, image_array):
    """Generate the Grad-CAM heatmap from a model wrapper and preprocessed image"""
    model = model_wrapper.model
    last_conv_layer = model_wrapper.get_last_conv_layer()

    # Use the layer name for compatibility
    conv_layer_name = last_conv_layer.name

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(conv_layer_name).output, model.output]
    )

    image_batch = np.expand_dims(image_array, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

    grads = tape.gradient(pred_output, conv_outputs)[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Multiply each channel by corresponding gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8

    # Resize heatmap
    heatmap = tf.image.resize(
        tf.expand_dims(heatmap, axis=-1),
        (224, 224)
    ).numpy().squeeze()

    return heatmap
def generate_saliency_map(model, image_array):
    image_tensor = tf.convert_to_tensor(np.expand_dims(image_array, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model.model(image_tensor)  # ✅ fix is here
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    gradients = tape.gradient(loss, image_tensor)[0]
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array.astype("uint8"))
    plt.axis('off')
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='hot')
    plt.axis('off')
    plt.title("Saliency Map")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    saliency_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return saliency_base64
