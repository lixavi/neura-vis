import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, image):
    # Convert image to array and preprocess it
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0
    
    # Get the last convolutional layer and the model's prediction
    last_conv_layer = model.get_layer('conv2d')  # Assuming the last convolutional layer is named 'conv2d'
    preds = model.predict(image_array)
    pred_class = np.argmax(preds[0])
    pred_output = model.output[:, pred_class]
    
    # Compute gradients of the predicted class with respect to the output feature map
    grads = tf.GradientTape().gradient(pred_output, last_conv_layer.output)[0]
    
    # Compute global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate heatmap
    heatmap = tf.reduce_mean(last_conv_layer.output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap *= pooled_grads.numpy()
    
    # Resize heatmap to match the original image size
    heatmap = tf.image.resize(heatmap, (image_array.shape[1], image_array.shape[2]))
    heatmap = heatmap.numpy()
    
    # Superimpose heatmap on the original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image_array[0], 0.6, heatmap, 0.4, 0)
    
    return superimposed_img
