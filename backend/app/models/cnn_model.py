import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4, efficientnet
import numpy as np

class SkinCancerModel:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.classes = [
            'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
            'melanoma', 'nevus', 'pigmented benign keratosis',
            'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
        ]
        
        try:
            self.model = tf.keras.models.load_model('best_model.h5')
            print("✅ Loaded pre-trained model successfully")
        except:
            print("🚧 No pre-trained model found, building new model")
            self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = efficientnet.preprocess_input(inputs)

        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )

        # Freeze all layers initially
        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(len(self.classes), activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        return model

    def fine_tune_model(self, unfreeze_from=400):
        """Unfreeze layers from a given index for fine-tuning"""
        print(f"🔓 Unfreezing layers from index {unfreeze_from} for fine-tuning...")
        self.model.get_layer('efficientnetb4').trainable = True
        for layer in self.model.get_layer('efficientnetb4').layers[:unfreeze_from]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

    def train(self, train_data, validation_data, epochs=20, fine_tune=False):
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])

        train_data = train_data.map(lambda x, y: (data_augmentation(x), y))

        history = self.model.fit(
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

        if fine_tune:
            self.fine_tune_model()
            print("🔁 Fine-tuning model...")
            self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs // 2,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        'best_model_finetuned.h5',
                        save_best_only=True,
                        monitor='val_accuracy'
                    )
                ]
            )

        return history

    def predict(self, image):
        processed_image = tf.image.resize(image, self.input_shape[:2])
        processed_image = tf.expand_dims(processed_image, 0)
        predictions = self.model.predict(processed_image)
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
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        return self.model.get_layer('efficientnetb4').layers[-1]
