## Neura Vis
Toolkit for neural network visualization, utilizing TensorFlow and Keras for model interpretation.

## Features
- Model Visualization: Visualize layer activations and filters.
- Interpretation Techniques: Implement Grad-CAM for model interpretation.
- Data Handling: Load and preprocess data efficiently.
- Image Utilities: Load and visualize images easily.

## Usage
- Install necessary dependencies.
- Define and train your neural network model.
- Use NeuraVis to visualize layer activations, filters, and perform model interpretation.

## Example

```python
from models.model import NeuralNetwork
from visualization.activations import visualize_activations
from utils.data_utils import load_data, preprocess_data

# Load and preprocess data
data = load_data()
preprocessed_data = preprocess_data(data)

# Define and train neural network model
model = NeuralNetwork().build_model()
model.fit(preprocessed_data, labels)

# Visualize layer activations
visualize_activations(model, preprocessed_data)

```

## Dependencies
```
TensorFlow
Keras
scikit-learn
matplotlib
OpenCV
```
