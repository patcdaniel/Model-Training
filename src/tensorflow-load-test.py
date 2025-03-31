import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU is available!")
    for gpu in gpus:
        print(f" -> {gpu}")
else:
    print("❌ No GPU found.")