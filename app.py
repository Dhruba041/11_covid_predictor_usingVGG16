import streamlit as st
import keras
import numpy as np
#from normalize import process_norm
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import gdown
import tensorflow as tf


st.markdown(
    """
    <style>
    .stApp {
        background-color:  #EE82EE;  /* Violet */
        color: brown;   /* default text color */
    }

    h1 {
        color: #1f77b4;  /* blue title */
    }

    label {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label span p {
        color: brown !important;
        font-weight: 600;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<h1 style='text-align: center;'>Covid Detection App</h1>",
    unsafe_allow_html=True
)




MODEL_PATH = "VGG16_model.keras"

FILE_ID = "14j-D_JivCSrCogx7WFuhVtj1I3oUXVUm"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )

model = tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)   
    image = image.resize((128, 128))

    # Normalize

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    #img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)

    if predicted_index == 0:
        Prediction_text = "**COVID-19 Positive**"
        bg_color = "#ffebee"      # Light red
        border_color = "#d32f2f"  # Red
        text_color = "#b71c1c"    # Dark red
    elif predicted_index == 1:
        Prediction_text = "**Normal**"
        bg_color = "#e8f5e9"      # Light green
        border_color = "#388e3c"  # Green
        text_color = "#1b5e20"    # Dark green
    else:
        Prediction_text = "**Viral Pneumonia**"
        bg_color = "#fff3e0"      # Light orange
        border_color = "#f57c00"  # Orange
        text_color = "#e65100"    # Dark orange

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:15px;
            border-radius:10px;
            border:2px solid {border_color};
            color:{text_color};
            font-size:18px;
            font-weight:bold;
            text-align:center;
        ">
            The image is predicted to be: {Prediction_text}
        </div>
        """,
        unsafe_allow_html=True
    )


