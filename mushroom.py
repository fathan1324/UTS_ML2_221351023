import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import time

# CSS styling
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .title { color: #4CAF50; font-size:40px; text-align:center; }
    .subtitle { color: #555; font-size:18px; text-align:center; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini memprediksi apakah jamur **beracun** atau **bisa dimakan** berdasarkan ciri-cirinya.

# Judul
st.markdown('<p class="title">🍄 Aplikasi Klasifikasi Jamur</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Masukkan ciri-ciri jamur untuk memprediksi apakah jamur beracun atau bisa dimakan.</p>', unsafe_allow_html=True)

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

# Mapping kode agar readable
kode_to_nama = {
    'cap-shape': {'b': 'berbentuk lonceng', 'c': 'berbentuk kerucut', 'x': 'berbentuk cembung',
                  'f': 'berbentuk datar', 'k': 'berbentuk menonjol', 's': 'berbentuk cekung'},
    'cap-surface': {'f': 'berserat', 'g': 'beralur', 'y': 'bersisik', 's': 'halus'},
    'cap-color': {'n': 'cokelat', 'b': 'kuning pucat', 'c': 'cokelat kayu manis', 'g': 'abu-abu', 'r': 'hijau',
                  'p': 'merah muda', 'u': 'ungu', 'e': 'merah', 'w': 'putih', 'y': 'kuning'},
    'bruises': {'t': 'memar', 'f': 'tidak memar'},
    'odor': {'a': 'bau almond', 'l': 'bau anise', 'c': 'bau kreosot', 'y': 'bau amis', 'f': 'bau busuk',
             'm': 'bau apak', 'n': 'tidak berbau', 'p': 'bau menyengat', 's': 'bau pedas'},
    'gill-attachment': {'a': 'menempel', 'f': 'bebas'},
    'gill-spacing': {'c': 'rapat', 'w': 'jarang'},
    'gill-size': {'b': 'lebar', 'n': 'sempit'},
    'gill-color': {'k': 'hitam', 'n': 'cokelat', 'b': 'kuning pucat', 'h': 'cokelat tua', 'g': 'abu-abu',
                   'r': 'hijau', 'o': 'oranye', 'p': 'merah muda', 'u': 'ungu', 'e': 'merah', 'w': 'putih', 'y': 'kuning'},
    'stalk-shape': {'e': 'membesar', 't': 'mengecil'},
    'stalk-root': {'b': 'berumbi', 'c': 'berbentuk tongkat', 'e': 'sama', 'r': 'berakar', '?': 'tidak diketahui'},
    'stalk-surface-above-ring': {'f': 'berserat', 'y': 'bersisik', 'k': 'seperti sutra', 's': 'halus'},
    'stalk-surface-below-ring': {'f': 'berserat', 'y': 'bersisik', 'k': 'seperti sutra', 's': 'halus'},
    'stalk-color-above-ring': {'n': 'cokelat', 'b': 'kuning pucat', 'c': 'cokelat kayu manis', 'g': 'abu-abu',
                               'o': 'oranye', 'p': 'merah muda', 'e': 'merah', 'w': 'putih', 'y': 'kuning'},
    'stalk-color-below-ring': {'n': 'cokelat', 'b': 'kuning pucat', 'c': 'cokelat kayu manis', 'g': 'abu-abu',
                               'o': 'oranye', 'p': 'merah muda', 'e': 'merah', 'w': 'putih', 'y': 'kuning'},
    'veil-type': {'p': 'parsial'},
    'veil-color': {'n': 'cokelat', 'o': 'oranye', 'w': 'putih', 'y': 'kuning'},
    'ring-number': {'n': 'tidak ada', 'o': 'satu', 't': 'dua'},
    'ring-type': {'e': 'menghilang', 'f': 'melebar', 'l': 'besar', 'n': 'tidak ada', 'p': 'menggantung'},
    'spore-print-color': {'k': 'hitam', 'n': 'cokelat', 'b': 'kuning pucat', 'h': 'cokelat tua', 'r': 'hijau',
                          'o': 'oranye', 'u': 'ungu', 'w': 'putih', 'y': 'kuning'},
    'population': {'a': 'banyak', 'c': 'bergerombol', 'n': 'cukup banyak', 's': 'tersebar', 'v': 'beberapa', 'y': 'sendiri'},
    'habitat': {'g': 'padang rumput', 'l': 'daun', 'm': 'padang rumput luas', 'p': 'jalan setapak',
                'u': 'kota', 'w': 'tempat sampah', 'd': 'hutan'}
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
input_readable = {}
for feature, mapping in label_encoders.items():
    readable_options = [kode_to_nama[feature][k] for k in mapping.keys()]
    selected_readable = st.selectbox(f"{feature.replace('-', ' ').capitalize()}", readable_options)
    selected_code = [k for k, v in kode_to_nama[feature].items() if v == selected_readable][0]
    encoded_val = mapping[selected_code]
    user_input.append(encoded_val)
    input_readable[feature] = selected_readable

# Tampilkan tabel input
st.markdown("### Ringkasan Input")
st.table(pd.DataFrame([input_readable]))

# Tombol prediksi
if st.button("🔍 Prediksi"):
    with st.spinner('Sedang memproses prediksi...'):
        time.sleep(1.5)  # simulasi loading
        input_array = np.array([user_input], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_label_index = np.argmax(output)
        predicted_class = label_encoder.inverse_transform([predicted_label_index])[0]

    st.markdown("## Hasil Prediksi 🎯")
    if predicted_class == 'e':
        st.success("Jamur ini bisa dimakan ✅")
    else:
        st.error("Jamur ini beracun ⚠️")
