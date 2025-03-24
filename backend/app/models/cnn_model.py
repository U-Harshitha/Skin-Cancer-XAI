import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class SkinCancerModel:
    def __init__(self):
        self.classes = [
            'actinic keratosis',
            'basal cell carcinoma',
            'dermatofibroma',
            'melanoma',
            'nevus',
            'pigmented benign keratosis',
            'seborrheic keratosis',
            'squamous cell carcinoma',
            'vascular lesion'
        ]
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            
            # First Convolutional Block
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Second Convolutional Block
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Third Convolutional Block
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, train_data, validation_data, epochs=20):
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    save_best_only=True
                )
            ]
        )

    def predict(self, image):
        # Ensure image is in the correct format (224x224x3)
        processed_image = tf.image.resize(image, (224, 224))
        processed_image = tf.expand_dims(processed_image, 0)
        
        # Get prediction probabilities
        predictions = self.model.predict(processed_image)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        
        return [
            {
                'class': self.classes[idx],
                'probability': float(predictions[0][idx])
            }
            for idx in top_3_idx
        ] 