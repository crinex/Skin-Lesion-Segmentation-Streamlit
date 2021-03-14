import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread

def enhance(img):
    # reshape(1, 256, 256, 1)
    #sub = (model.predict(img.reshape(1,256,256,3))).flatten()
    img = img.reshape((1, 256, 256, 3)).astype(np.float32) / 255.
    sub = (model.predict(img)).flatten()

    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

def applyMask(img):
    sub = img.reshape((1, 256, 256, 3)).astype(np.float32) / 255.
    #sub = np.array(img.reshape(256, 256), dtype=np.uint8)
    mask = np.array(enhance(sub).reshape(256, 256), dtype=np.uint8)
    sub2 = img.reshape(256, 256, 3)
    #sub2 = np.array(img.reshape(256, 256, 3), dtype=np.uint8)
    res = cv2.bitwise_and(sub2, sub2, mask = mask)

    return res

##############
# Model Load #
##############
@st.cache
def load():
    return load_model('ResU_net.h5')
model = load()


##############
# Side Bar   #
##############
with st.sidebar.header('Upload your Skin Image'):
    upload_file = st.sidebar.file_uploader('Choose your Skin Image', type=['jpg', 'jpeg', 'png'])


##############
# Page Title #
##############
st.write('# 🧠Skin Lesion Segmentation🧠')
st.write('This Website was created by Crinex. The code for the Website and Segmentation is in the Github. If you want to use this Code, please Fork and use it.🤩🤩')
st.write('📕 Github:https://github.com/crinex/Skin-Lesion-Segmentation-Streamlit 📕')


###############
# Main Screen #
###############
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.write('### Original Image')
    img = imread(upload_file)
    img = resize(img, (256, 256))
    preview_img = resize(img, (256, 256))
    st.image(preview_img)

col2.write('### Button')
clicked = col2.button('Segment!!')
clicked2 = col2.button('Predict Image')

if clicked:
    x = img
    #x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #x = x.reshape((1, 256, 256, 3)).astype(np.float32) / 255.
    x = np.reshape(x, (256, 256, 3))
    #x = resize(x, (256, 256, 3))
    #pred = model.predict(x).squeeze()
    col3.write('### Segmentation Image')
    mask_img = applyMask(x)
    col3.image(mask_img)

if clicked2:
    x = img
    #x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = np.reshape(x, (256, 256, 3))
    #x = resize(x, (256, 256, 1))
    enhance_img = enhance(x).reshape(256, 256)
    col3.write('### Prediction Image')
    col3.image(enhance_img)
