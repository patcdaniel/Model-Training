import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
import os


def build_model(input_shape, num_classes):

    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='Adam',
        loss='sparse_categorical_crossentropy',  # ✅ Correct for integer targets
        metrics=['sparse_categorical_accuracy'],
    )
    return model, base_model


class FineTuneCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_model, unfreeze_epoch, learning_rate):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_epoch = unfreeze_epoch
        self.learning_rate = learning_rate
        self.fine_tuned = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.fine_tuned and epoch + 1 >= self.unfreeze_epoch:
            print(f"\n🔓 Unfreezing base model after epoch {epoch + 1}")
            self.base_model.trainable = True
            self.fine_tuned = True
            self._recompile_after_training = True
        else:
            self._recompile_after_training = False

    def on_train_end(self, logs=None):
        if getattr(self, "_recompile_after_training", False):
            print("♻️ Recompiling model for fine-tuning after training...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy']
            )


class PeriodicSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, every_n_epochs=5, prefix="manual_save"):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            save_path = os.path.join(self.output_dir, f"{self.prefix}_epoch_{epoch + 1}.keras")
            self.model.save(save_path)
            print(f"Mid-model saved to {save_path}")