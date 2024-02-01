import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_activations(model, data):
    activation_model = tf.keras.models.Model(inputs=model.input,
                                              outputs=[layer.output for layer in model.layers])
    
    activations = activation_model.predict(data)
    layer_names = [layer.name for layer in model.layers]

    for layer_name, layer_activation in zip(layer_names, activations):
        if len(layer_activation.shape) == 4:  # Check if the layer activation is from a convolutional layer
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features)
            
            # Display the feature maps for the first few features in this format (size, size, n_features)
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                activation = layer_activation[0, :, :, i]
                activation -= activation.mean()  # Post-process the feature to make it visually palatable
                activation /= activation.std()
                activation *= 64
                activation += 128
                activation = np.clip(activation, 0, 255).astype('uint8')
                display_grid[:, i * size : (i + 1) * size] = activation
                
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.show()
