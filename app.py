# app.py

import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

st.title("Reconnaissance Faciale")
st.write("Téléversez une image ou prenez une photo pour reconnaître la personne.")

# --- Charger les modèles ---
@st.cache_resource
def load_models():
    svm_model = pickle.load(open('svm_model_facenet.pkl', 'rb'))
    encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    facenet_model = FaceNet()
    return svm_model, encoder, facenet_model

svm_model, encoder, facenet_model = load_models()

# --- Détecter visage ---
def extract_face(image, required_size=(160,160)):
    detector = MTCNN()
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face

# --- Obtenir embedding ---
def get_embedding(face):
    emb = facenet_model.embeddings([face])[0]
    emb = emb / np.linalg.norm(emb)
    return emb

# --- Choix image ---
choice = st.radio("Choisir la source :", ("Téléverser une image", "Prendre une photo"))

img = None
if choice == "Téléverser une image":
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
elif choice == "Prendre une photo":
    img_file_buffer = st.camera_input("Prendre une photo")
    if img_file_buffer is not None:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Si image existe ---
if img is not None:
    st.image(img, caption="Image sélectionnée", use_column_width=True)
    face = extract_face(img)
    if face is not None:
        embedding = get_embedding(face).reshape(1, -1)
        pred_label = encoder.inverse_transform(svm_model.predict(embedding))[0]
        st.success(f"🔹 Prédiction : {pred_label}")
    else:
        st.error("Aucun visage détecté.")
