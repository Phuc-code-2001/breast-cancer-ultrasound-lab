import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import secrets

IMG_SIZE = 224

model : keras.Model = keras.models.load_model(r'.\api\cnn-models\busi_Proposed_ph2_model.h5')
model.trainable = False
model.summary()

classes = ['benign', 'normal', 'malignant']

def predict(file):

    bytes = file.read()
    img = cv2.imdecode(np.frombuffer(bytes , np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    x = np.expand_dims(img, axis=0)
    probs = model.predict(x)[0]
    
    return { classes[i]: prob for i, prob in enumerate(probs) }

