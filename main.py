import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("Fundbüro KI – Kleidungserkennung")

st.write("Lade ein Bild hoch. Die KI erkennt Objekte im Bild.")

# YOLO Modell laden (vortrainiert)
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # kleines schnelles Modell
    return model

model = load_model()

# Bild Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    if st.button("Erkennung starten"):

        # Bild in numpy umwandeln
        img_array = np.array(image)

        # YOLO Prediction
        results = model(img_array)

        # Annotiertes Bild erzeugen
        annotated_frame = results[0].plot()

        # BGR -> RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        st.subheader("Erkannte Objekte")
        st.image(annotated_frame, use_container_width=True)