import matplotlib.pyplot as plt
import numpy as np

def visualize_filters(model):
    for layer in model.layers:
        if 'conv' in layer.name:  # Check if the layer is a convolutional layer
            filters, biases = layer.get_weights()
            n_filters = filters.shape[3]  # Number of filters in the layer
            
            # Calculate the number of columns for the subplots
            n_cols = min(n_filters, 6)
            n_rows = (n_filters // n_cols) + (1 if n_filters % n_cols != 0 else 0)
            
            # Plot each filter
            plt.figure(figsize=(10, 10))
            for i in range(n_filters):
                # Get the weights of the i-th filter
                filt = filters[:, :, :, i]
                
                # Normalize the filter weights to [0, 1]
                f_min, f_max = filt.min(), filt.max()
                filt = (filt - f_min) / (f_max - f_min)
                
                # Plot the filter
                plt.subplot(n_rows, n_cols, i+1)
                plt.imshow(filt[:, :, 0], cmap='gray')
                plt.axis('off')
            plt.suptitle(layer.name, fontsize=16)
            plt.show()
