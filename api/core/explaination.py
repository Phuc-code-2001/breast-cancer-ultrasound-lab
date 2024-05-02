import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import threading
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from django.conf import settings
from secrets import token_hex

class GradCAM:
    
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName
            
        if self.layerName == None:
            self.layerName = self.find_target_layer()
    
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")
            
    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        
        gradModel = tf.keras.Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.output],
            name='gradcam'
        )
        
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)
            loss = preds[:, classIdx]
        
        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        
        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        
        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size)
        
        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        
        return cam3


def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    new_img = 0.5 * cam3 + 0.5 * img
    return (new_img * 255.0 / new_img.max()).astype("uint8")


IMG_SIZE = 224
classes = ['benign', 'normal', 'malignant']
media_url = settings.MEDIA_URL
media_root = settings.MEDIA_ROOT

grad_output = dict()
thread_manager = dict()

def save_gradCAMs_async(id, gradCAM, x, ground_truth, probs):
    
    plt.figure(figsize=(10, 5))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(x[0])
    plt.title("Input Image")
    plt.axis("off")
    
    # Show overlayed grad
    plt.subplot(1, 2, 2)
    idx = np.argmax(probs)
    confidence = np.max(probs)
    heat_map = gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=(IMG_SIZE, IMG_SIZE))
    overlay_img = overlay_gradCAM(x[0], heat_map)
    plt.imshow(overlay_img)
    plt.title("Predict: {}\nConfidence: {:.4f}".format(classes[idx], confidence))
    plt.axis("off")
    
    foldpath = media_root
    os.makedirs(foldpath, exist_ok=True)
    savepath = os.path.join(foldpath, f'{id}.png')
    plt.savefig(savepath, dpi=72, bbox_inches='tight')
    grad_output[id] = f'{media_url}/{id}.png'
    

def create_gradcam(gradCam, x, ground_truth, probs):
    
    id = token_hex(16)
    thread = threading.Thread(target=save_gradCAMs_async, args=(id, gradCam, x, ground_truth, probs))
    thread_manager[id] = thread
    thread.start()
    return id

def get_gradcam_url(id):
    return grad_output.get(id)