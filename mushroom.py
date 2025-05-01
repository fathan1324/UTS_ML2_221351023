import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Mapping encoding hasil LabelEncoder di training
label_encoders = {
    'cap-shape': {'b': 0, 'c': 1, 'f': 2, 'k': 3, 's': 4, 'x': 5},
    'cap-surface': {'f': 0, 'g': 1, 's': 2, 'y': 3},
    'cap-color': {'b': 0, 'c': 1, 'e': 2, 'g': 3, 'n': 4, 'p': 5, 'r': 6, 'u': 7, 'w': 8, 'y': 9},
    'bruises': {'f': 0, 't': 1},
    'odor': {'a': 0, 'c': 1, 'f': 2, 'l': 3, 'm': 4, 'n': 5, 'p': 6, 's': 7, 'y': 8},
    'gill-attachment': {'a': 0, 'f': 1},
    'gill-spacing': {'c': 0, 'w': 1},
    'gill-size': {'b': 0, 'n': 1},
    'gill-color': {'b': 0, 'e': 1, 'g': 2, 'h': 3, 'k': 4, 'n': 5, 'o': 6, 'p': 7, 'r': 8, 'u': 9, 'w': 10, 'y': 11},
    'stalk-shape': {'e': 0, 't': 1},
    'stalk-root': {'?': 0, 'b': 1, 'c': 2, 'e': 3, 'r': 4},
    'stalk-surface-above-ring': {'f': 0, 'k': 1, 's': 2, 'y': 3},
    'stalk-surface-below-ring': {'f': 0, 'k': 1, 's': 2, 'y': 3},
    'stalk-color-above-ring': {'b': 0, 'c': 1, 'e': 2, 'g': 3, 'n': 4, 'o': 5, 'p': 6, 'w': 7, 'y': 8},
    'stalk-color-below-ring': {'b': 0, 'c': 1, 'e': 2, 'g': 3, 'n': 4, 'o': 5, 'p': 6, 'w': 7, 'y': 8},
    'veil-type': {'p': 0},
    'veil-color': {'n': 0, 'o': 1, 'w': 2, 'y': 3},
    'ring-number': {'n': 0, 'o': 1, 't': 2},
    'ring-type': {'e': 0, 'f': 1, 'l': 2, 'n': 3, 'p': 4},
    'spore-print-color': {'b': 0, 'h': 1, 'k': 2, 'n': 3, 'o': 4, 'r': 5, 'u': 6, 'w': 7, 'y': 8},
    'population': {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
    'habitat': {'d': 0, 'g': 1, 'l': 2, 'm': 3, 'p': 4, 'u': 5, 'w': 6}
}

# Load label encoder untuk output
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="mushroom_ann.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Mushroom Classification App 🍄")
st.write("Masukkan ciri-ciri jamur, lalu sistem akan memprediksi apakah jamur **beracun** atau **bisa dimakan**.")

# Form input
user_input = []
for feature, mapping in label_encoders.items():
    option = st.selectbox(f"{feature.replace('-', ' ').capitalize()}", list(mapping.keys()))
    encoded_val = mapping[option]
    user_input.append(encoded_val)

if st.button("Prediksi"):
    input_array = np.array([user_input], dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    # Ambil output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_label_index = np.argmax(output)
    predicted_class = label_encoder.inverse_transform([predicted_label_index])[0]

    if predicted_class == 'e':
        st.success("Jamur ini bisa dimakan (edible)")
    else:
        st.error("Jamur ini beracun (poisonous)")
