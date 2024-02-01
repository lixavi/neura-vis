import matplotlib.pyplot as plt
import cv2

def load_image(image_path):
    # Load image from file
    image = cv2.imread(image_path)
    # Convert image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def visualize_image(image, title="Image"):
    # Display the image
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
