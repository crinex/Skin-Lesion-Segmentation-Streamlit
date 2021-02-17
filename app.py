import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread

def enhance(img):
    sub = (model.predict(img.reshape(1,256,256,1))).flatten()

    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

def applyMask(img):
    sub = np.array(img.reshape(256, 256), dtype=np.uint8)
    mask = np.array(enhance(sub).reshape(256, 256), dtype=np.uint8)
    res = cv2.bitwise_and(sub, sub, mask = mask)

    return res

##############
# Model Load #
##############
@st.cache
def load():
    return load_model('SegNet.h5')
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


###############
# Main Screen #
###############
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.write('### Original Image')
    img = imread(upload_file)
    img = cv2.resize(img, dsize=(256, 256))
    preview_img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    st.image(preview_img)

col2.write('### Button')
clicked = col2.button('Segment!!')
clicked2 = col2.button('Predict Image')

if clicked:
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = x.reshape((1, 256, 256, 1))
    pred = model.predict(x).squeeze()
    col3.write('### Segmentation Image')
    mask_img = applyMask(x)
    col3.image(mask_img)

if clicked2:
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = x.reshape((1, 256, 256, 1))
    enhance_img = enhance(x).reshape(256, 256).squeeze()
    col3.write('### Prediction Image')
    col3.image(enhance_img)
