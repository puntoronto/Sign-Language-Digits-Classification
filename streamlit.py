import streamlit as st  
import tensorflow as tf
import numpy as np
import PIL
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# kesif sayfasini hazirlama
def show_explore_page():
    st.title('Explore Page')
    st.write('''There are 15000 photos in our dataset. It contains american sign language numbers from 0 to 9. We used 12000 photos for training and 3000 photos for testing. 
    We used 100x100 grayscale images for training. We used 4 convolutional layers and 4 max pooling layers for training. We used 3 dense layers for training. We used 0.5 dropout rate for training. 
    We used Adam optimizer for training. We used categorical crossentropy loss function for training. We used accuracy metric for training. We used 10 epochs for training. We used 32 batch size for training. We used 0.2 validation split for training.''')
    img_path_dict = {'Examples from Dataset': 'https://github.com/yektaozan/Veri-Siniflandirma-Odev/blob/main/sample.png',
                        'Classification Report': 'https://github.com/yektaozan/Veri-Siniflandirma-Odev/blob/main/sign_language_model_classification_report.png',
                        'Confusion Matrix': 'https://github.com/yektaozan/Veri-Siniflandirma-Odev/blob/main/sign_language_model_confusion_matrix.png'}
    
    for img_name, img_path in img_path_dict.items():
        st.write(img_name)
        img = PIL.Image.open(img_path).resize((900, 900))
        st.image(img)
        st.write('#'*50)
    
# modeli yukleme
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# tahmin etme   
def predict_sign_language(model, img_array):
    prediction = model.predict(img_array)
    return str(np.argmax(prediction, axis=1)[0])

# fotograflari tahmin etme
def prepare(img_path, model):
    img_size = 100
    img = load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array/255
    img_array = np.expand_dims(img_array, axis=0)
    prediction = predict_sign_language(model, img_array)
    return prediction

# tahmin sayfasini hazirlama
def show_predict_page():
    st.title('Prediction Page')
    model_path = "https://github.com/yektaozan/Veri-Siniflandirma-Odev/blob/main/isaret_dili_model.h5"
    model = load_model(model_path)
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        img = PIL.Image.open(uploaded_file).resize((300, 300))
        st.image(img, caption='Uploaded Image.')
        prediction = prepare(uploaded_file.name, model)
        st.write('Prediction: {}'.format(prediction))
        st.write('Done!')

page = st.sidebar.selectbox("Explore Or Predict", ("Explore", "Predict"))

if page == 'Explore':
    show_explore_page()
else:
    show_predict_page()