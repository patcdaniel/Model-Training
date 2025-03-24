import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


class FineTuneCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_model, unfreeze_epoch, learning_rate):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_epoch = unfreeze_epoch
        self.learning_rate = learning_rate
        self.fine_tuned = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self.fine_tuned and epoch >= self.unfreeze_epoch:
            print(f"\n🔓 Unfreezing base model at epoch {epoch}")
            self.base_model.trainable = True
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            self.fine_tuned = True