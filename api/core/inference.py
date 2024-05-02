import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import cv2
import numpy as np

from .explaination import GradCAM, create_gradcam

IMG_SIZE = 224

model : keras.Model = keras.models.load_model(r'.\api\cnn-models\busi_Proposed_ph2_model.h5')
model.trainable = False
model.summary()

gradcam = GradCAM(model)

classes = ['benign', 'normal', 'malignant']

def predict(file):

    bytes = file.read()
    img = cv2.imdecode(np.frombuffer(bytes , np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    x = np.expand_dims(img, axis=0)
    probs = model.predict(x)[0]
    label_id = np.argmax(probs)
    
    gradcam_id = create_gradcam(gradcam, x, "Unknown", probs)
    
    results = {
        'label': classes[label_id],
        'confidence': np.max(probs),
        'probs': { classes[i]: prob for i, prob in enumerate(probs) },
        'gradcam_id': gradcam_id
    }
    
    return results

