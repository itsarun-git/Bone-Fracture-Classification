
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import os

#load pretrained model (ensure it is in same directory as streamlit app file)
model_path = os.path.join(os.path.dirname(__file__), "fracture_classifier_final.keras")
model = load_model(model_path)

#class names
class_names = ['Avulsion fracture',
 'Comminuted fracture',
 'Fracture Dislocation',
 'Greenstick fracture',
 'Hairline Fracture',
 'Impacted fracture',
 'Longitudinal fracture',
 'Oblique fracture',
 'Pathological fracture',
 'Spiral Fracture']

img_size = 128

#function to run prediction
def run_prediction(image_path, model):

    image = Image.open(image_path).convert('RGB')
    #resize to match models input shape
    image = image.resize((img_size,img_size)) #shape => (128, 128, 3) 

    #convert image 
    image_array = np.array(image)
    #expand as the input layer expects batch size as well
    image_array = np.expand_dims(image_array, axis = 0)  #shape => (1,128,128,3)

    #since using resnet pretrained model where we used (weights='imagenet)
    image_array = preprocess_input(image_array)

    preds = model.predict(image_array)

    predicted_class = np.argmax(preds, axis=1)[0] #max label
    confidence_score = preds[0][predicted_class]*100
    predicted_label = class_names[predicted_class]

    return predicted_label, confidence_score

uploaded_image = st.file_uploader('Upload an X-ray image of fracture for classification', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    image_path = './temp_image.jpg'

    with open(image_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())
    
    
    label, confidence = run_prediction(image_path, model)
    #to view the uploaded image
    image = Image.open(image_path).convert('RGB').resize((518, 518))
    st.image(image, caption='Upoloaded X-Ray Image', use_container_width= True)
    st.success(f"Prediction: **{label}** with **{confidence:.2f}%** confidence")


