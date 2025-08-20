import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("ladybird_model.h5")

# Convert it
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save it
with open("ladybird_model.tflite", "wb") as f:
    f.write(tflite_model)