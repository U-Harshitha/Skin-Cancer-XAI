import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4, efficientnet
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
        try:
            # Try to load pre-trained model
            self.model = tf.keras.models.load_model('best_model.h5')
            print("Loaded pre-trained model successfully")
        except:
            print("No pre-trained model found, building new model")
            self.model = self.build_model()

    def build_model(self):
        # Create input layer with preprocessing
        inputs = layers.Input(shape=(224, 224, 3))
        x = efficientnet.preprocess_input(inputs)
        
        # Use EfficientNetB4 as base model
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        
        # Fine-tune the last few layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Add custom layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(len(self.classes), activation='softmax')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model

    def train(self, train_data, validation_data, epochs=50):
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.2),
            layers.RandomFlip("horizontal"),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])

        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
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
        
        return {
            'predictions': [
                {
                    'class': self.classes[idx],
                    'probability': float(predictions[0][idx])
                }
                for idx in top_3_idx
            ]
        }

    def get_last_conv_layer(self):
        """Get the last convolutional layer of the base model"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        return self.model.get_layer('efficientnetb4').layers[-1] 