from models.model import NeuralNetwork
from visualization.activations import visualize_activations
from visualization.filters import visualize_filters
from visualization.gradcam import generate_gradcam
from utils.data_utils import load_data, preprocess_data
from utils.image_utils import load_image, visualize_image

def main():
    # Example usage of NeuraVis toolkit
    model = NeuralNetwork().build_model()
    data = load_data()
    preprocessed_data = preprocess_data(data)
    image = load_image()
    
    # Visualize layer activations
    print("Visualizing layer activations...")
    visualize_activations(model, preprocessed_data)
    
    # Visualize filters
    print("Visualizing filters...")
    visualize_filters(model)
    
    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    gradcam_image = generate_gradcam(model, image)
    visualize_image(gradcam_image, title="Grad-CAM")
    
    # Visualize original image
    print("Visualizing original image...")
    visualize_image(image, title="Original Image")
    

if __name__ == "__main__":
    main()
