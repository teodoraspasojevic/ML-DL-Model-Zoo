import os
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.segmentation import slic
from skimage.color import label2rgb
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cosine


class LIME:
    """
    Local Interpretable Model-Agnostic (LIME) class.

    This class implements the LIME algorithm for image classification models.

    Attributes:
        model (torch.nn.Module): The pre-trained image classification model.
        image_path (str): The path to the input image.
        image (np.ndarray): The loaded image.
        segments (np.ndarray): The segmented image.
        perturbations (list): List of perturbations applied to the image.
        predictions (np.ndarray): Model predictions for each perturbation.
        top_classes (list): List of top predicted classes for the image.
        self.similarity_weights(list): List of weights measuring similarity between original and all perturbed images.
        weights (np.ndarray): Weights for each superpixel in the image.
        self.surrogate_model(sklearn.linear_model): Surrogate model.
        labels (dict): Dictionary mapping class indices to human-readable labels.
    """

    def __init__(self, model, image_path):
        """
        Args:
            model (torch.nn.Module): Pre-trained image classification model.
            image_path (str): Path to the input image.
        """
        self.model = model
        self.image_path = image_path
        self.image = self.load_image(self.image_path)
        self.segments = None
        self.perturbations = None
        self.predictions = None
        self.top_classes = None
        self.similarity_weights = None
        self.weights = None
        self.surrogate_model = None

        with open('./data/imagenet-simple-labels.json') as f:
            self.labels = json.load(f)

    def load_image(self, image_path):
        """
        Loads an image from the specified path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            PIL.Image.Image: Loaded image.
        """
        image = Image.open(image_path).convert('RGB')
        return image

    def segment_image(self, image, n_segments=20):
        """
        Segments the image into superpixels.

        Args:
            image (PIL.Image.Image): Input image.
            n_segments (int): Number of segments for the slic algorithm.

        Returns:
            np.ndarray: Segmented image.
        """
        segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)
        return segments

    def perturb_image(self, image, segments, num_perturb=50):
        """
        Generates perturbed images by masking superpixels.

        Args:
            image (PIL.Image.Image): Input image.
            segments (np.ndarray): Segmented image.
            num_perturb (int): Number of perturbed images to generate.

        Returns:
            np.ndarray: Array of perturbations.
            list: List of perturbed images.
        """
        active_segments = np.unique(segments)
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, len(active_segments)))
        perturbed_images = np.tile(image, (num_perturb, 1, 1, 1))
        similarity_weights = []
        for i in range(num_perturb):
            for j, active in enumerate(active_segments):
                if perturbations[i, j] == 0:
                    perturbed_images[i][segments == active] = 0
            similarity = 1 - cosine(np.array(image).flatten().reshape(-1,1), perturbed_images[i].flatten().reshape(-1, 1))
            similarity_weights.append(similarity)
        perturbed_images_pil = [Image.fromarray(img.astype('uint8')) for img in perturbed_images]
        self.similarity_weights = similarity_weights
        return perturbations, perturbed_images_pil

    def transform_image(self, image):
        """
        Preprocesses the image for the model input.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # We use values from ImageNet dataset.
        ])
        return transform(image).unsqueeze(0)

    def get_top_predictions(self, prediction, top_n=2):
        """
        Gets the top N predictions from the model output.

        Args:
            prediction (torch.Tensor): Model prediction output.
            top_n (int): Number of top predictions to return.

        Returns:
            list: List of top N class indices.
        """
        probs = torch.nn.functional.softmax(prediction, dim=1)
        top_probs, top_classes = torch.topk(probs, top_n)
        return top_classes.squeeze().tolist()

    def find_params(self, num_segments=10, num_perturbations=5, top_n=2):
        """
        Finds the parameters and predictions for LIME.

        Args:
            num_segments (int): Number of segments for the slic algorithm.
            num_perturbations (int): Number of perturbed images to generate.
            top_n (int): Number of top predictions to consider.
        """
        # Segment the image.
        segments = self.segment_image(self.image, n_segments=num_segments)
        self.segments = segments

        # Perturb the image into num_perturbations new images, by activating random segments only.
        perturbations, perturbed_images = self.perturb_image(self.image, self.segments, num_perturb=num_perturbations)
        self.perturbations = perturbations

        # Get the predictions of the model on the original image.
        prediction = self.model(self.transform_image(self.image))
        self.top_classes = self.get_top_predictions(prediction, top_n=top_n)
        self.top_labels = [self.labels[cl] for cl in self.top_classes]

        # Get the predictions of the model on new images.
        predictions = []
        for image in perturbed_images:
            transformed_image = self.transform_image(image)
            prediction = self.model(transformed_image)
            predictions.append(prediction)
        self.predictions = predictions

        # Plot the original and perturbed image.
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title('Original image')

        plt.subplot(1, 2, 2)
        plt.imshow(perturbed_images[0])
        plt.title('Perturbed image')
        plt.show()

    def fit_local_surrogate_model(self, perturbations, predictions, top_class_index):
        """
        Fits a Linear regression model to get the local surrogate model weights.

        Args:
            perturbations (np.ndarray): Array of perturbations.
            predictions (list): List of model predictions on perturbed images.
            top_class_index (int): Index of the top class to explain.

        Returns:
            np.ndarray: local_surrogate weights for the superpixels.
        """
        y = np.array([pred.squeeze().tolist()[top_class_index] for pred in predictions])
        weights = np.array(self.similarity_weights)
        self.surrogate_model = LinearRegression()
        self.surrogate_model.fit(perturbations, y, sample_weight=weights)
        return self.surrogate_model.coef_

    def visualize_local_surrogate_model(self, image, segments, weights, i=0):
        """
        Visualizes the local surrogate model by only displaying superpixels that influence mostly the prediction.

        Args:
            image (np.ndarray): Input image.
            segments (np.ndarray): Segmented image.
            weights (np.ndarray): local_surrogate weights for the superpixels.
            i (int): Index for subplot.
        """
        # Convert PIL Image to NumPy array.
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Identify the top segments with the highest weights.
        top_segments = np.argsort(weights)[-25:]

        # Create a mask for the top segments.
        mask = np.isin(segments, top_segments)  # assuming segment labels start from 1

        # Create an all-black image
        new_image = np.zeros_like(image)

        # Apply the mask to retain original pixels in top segments.
        new_image[mask] = image[mask]

        # Display the local surrogate model.
        plt.imshow(new_image)
        plt.title(f"LIME local_surrogate for top{i+1}-prediction. Prediction label: {self.top_labels[i]}")
        plt.axis('off')
        plt.show()

    def fit(self, top_n=2):
        """
        Fits the local surrogate model for the top N predictions and visualizes the results.

        Args:
            top_n (int): Number of top predictions to explain.
        """
        lime_weights = []
        for i in range(top_n):
            top_class_index = self.top_classes[i]
            weights = self.fit_local_surrogate_model(self.perturbations, self.predictions, top_class_index)
            lime_weights.append(weights)

            self.visualize_local_surrogate_model(self.image, self.segments, weights, i=i)

        self.weights = lime_weights


if __name__ == '__main__':

    model = models.inception_v3(pretrained=True)
    model.eval()

    image_dir = './data/'

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):

            image_path = os.path.join(image_dir, filename)

            lime = LIME(model, image_path)
            lime.find_params(num_segments=100, num_perturbations=100, top_n=2)

            lime.fit(top_n=2)
