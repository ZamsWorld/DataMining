import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
import seaborn as sns

# Menambahkan CSS untuk mempercantik tampilan dengan tema merah, putih, dan hitam
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: #121212; /* Warna latar belakang gelap */
    }
    .stApp {
        background-color: #1c1c1c; /* Latar belakang utama hitam */
        padding: 40px 30px;
        border-radius: 15px;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
        margin-top: 50px;
    }
    h1, h2, h3, h4 {
        color: #FFFFFF; /* Teks putih untuk judul */
        font-weight: 600;
    }
    .stButton button {
        background-color: #FF4C4C; /* Merah untuk tombol */
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #D13434; /* Merah gelap saat hover */
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        font-size: 14px;
        text-align: left;
        color: #FFFFFF; /* Teks putih untuk tabel */
    }
    th {
        background-color: #FF4C4C; /* Merah untuk header tabel */
        color: white;
        text-align: center;
        padding: 12px 15px;
        border-radius: 10px;
    }
    td {
        border: 1px solid #444444;
        padding: 12px 15px;
    }
    tr:nth-child(even) {
        background-color: #333333; /* Warna baris genap lebih gelap */
    }
    tr:hover {
        background-color: #444444; /* Highlight baris tabel saat hover */
    }
    .stFileUploader {
        background-color: #FF4C4C;
        border-radius: 8px;
        padding: 10px 15px;
        color: white;
        font-weight: bold;
        cursor: pointer;
        text-align: center;
    }
    .stFileUploader:hover {
        background-color: #D13434;
    }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>Aplikasi Klasifikasi Obat</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FF4C4C;'>Menggunakan Algoritma Decision Tree</h3>", unsafe_allow_html=True)

# Memuat dataset dari file CSV
data = pd.read_csv('Classification.csv')

# Menampilkan dataset
st.markdown("<h3>Dataset</h3>", unsafe_allow_html=True)
st.dataframe(data)

# Memisahkan fitur dan target
X = data.drop("Drug", axis=1)
y = data["Drug"]

# Mengubah data kategoris menjadi numerik
X = pd.get_dummies(X, columns=["Sex", "BP", "Cholesterol"], drop_first=True)

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred = model.predict(X_test)

# Menampilkan hasil evaluasi
st.markdown("<h3>Hasil Evaluasi Model</h3>", unsafe_allow_html=True)
st.write(f"**Akurasi Model:** {accuracy_score(y_test, y_pred):.2f}")
st.markdown("<h4>Classification Report:</h4>", unsafe_allow_html=True)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.table(report_df)

# Visualisasi Decision Tree
st.markdown("<h3>Visualisasi Decision Tree</h3>", unsafe_allow_html=True)
fig = plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
st.pyplot(fig)

# Menampilkan Feature Importance
st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
st.table(feature_importance)

# Visualisasi Feature Importance
st.markdown("<h3>Visualisasi Feature Importance</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
st.pyplot(fig)

# Input data baru untuk prediksi
st.markdown("<h3>Coba Prediksi Data Baru</h3>", unsafe_allow_html=True)
age = st.slider("Umur", min_value=0, max_value=120, value=30)
na_to_k = st.slider("Rasio Natrium terhadap Kalium", min_value=0.0, max_value=100.0, value=15.0)
sex = st.selectbox("Jenis Kelamin", options=["M", "F"])
bp = st.selectbox("Tekanan Darah", options=["LOW", "NORMAL", "HIGH"])
cholesterol = st.selectbox("Kolesterol", options=["NORMAL", "HIGH"])

# Menyusun data input
input_data = pd.DataFrame({
    "Age": [age],
    "Na_to_K": [na_to_k],
    "Sex_M": [1 if sex == "M" else 0],
    "BP_LOW": [1 if bp == "LOW" else 0],
    "BP_NORMAL": [1 if bp == "NORMAL" else 0],
    "Cholesterol_NORMAL": [1 if cholesterol == "NORMAL" else 0],
})

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    st.success(f"Prediksi Obat: **{prediction[0]}**")

    # Fitur Download Hasil Prediksi
    result = pd.DataFrame(input_data)
    result['Prediksi'] = prediction
    st.download_button(
        label="Download Hasil Prediksi",
        data=result.to_csv(index=False),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )
