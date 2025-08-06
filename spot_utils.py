"""Utility functions for processing parking spot data and images.

This module provides functions to extract metadata from image paths,
process XML files containing parking spot information, and extract HOG features
from images for the SVM model.
"""
import os

import cv2

import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from imutils import perspective
from skimage.feature import hog


def extract_image_metadata(image_path: str) -> dict:
    """
    Extract camera name, weather, and date information from an image path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing camera_name, weather, and date
    """
    metadata = {
        'camera_name': 'Unknown',
        'weather': 'Unknown',
        'date': 'Unknown'
    }
    
    try:
        path_parts = image_path.split('/')
        
        # Find camera name and its position
        camera_names = ['PUCPR', 'UFPR04', 'UFPR05']
        weather_types = ['Cloudy', 'Rainy', 'Sunny']
        
        for j, part in enumerate(path_parts):
            if part in camera_names:
                metadata['camera_name'] = part
                
                # Check for weather in the next part
                if j+1 < len(path_parts) and path_parts[j+1] in weather_types:
                    metadata['weather'] = path_parts[j+1]
                
                # Check for date in the part after weather
                if j+2 < len(path_parts) and path_parts[j+2].startswith('20'):
                    metadata['date'] = path_parts[j+2]
                
                break  # Found what we needed, exit the loop
                
    except Exception as e:
        print(f'Error extracting metadata from path {image_path}: {str(e)}')
    
    return metadata


def get_valid_spaces(root: ET.Element) -> list[tuple[ET.Element, list[list[int]]]]:
    """
    Extracts valid parking spaces from the XML root element.

    Args:
        root (ET.Element): The root element of the XML tree.
    
    Returns:
        list[tuple[ET.Element, list[list[int]]]]: List of tuples where each tuple contains:
            - ET.Element: The space element.
            - list[list[int]]: List of points defining the contour of the parking space.
    """
    valid_spaces = []
    for space in root.findall('space'):
        # Skip spaces without an 'occupied' attribute or with None value
        if space.get('occupied') is None:
            continue
        
        # Get the contour points for the space
        contour = space.find('contour')
        if contour is None:
            print(f'No contour found for space ID {space.get("id")}')
            continue
        
        points = []
        # Extract points from the contour
        for p in contour.findall('*'):
            # Some XML files use 'point' tags, others use 'Point'
            if p.tag.lower() == 'point':
                x_val = p.get('x')
                y_val = p.get('y')
                if x_val is not None and y_val is not None:
                    points.append([int(x_val), int(y_val)])
        
        if not points:
            print(f'No points found in contour for space ID {space.get("id")}')
            continue
        
        valid_spaces.append((space, points))
    
    return valid_spaces


def get_spot_data_from_xml(xml_path: str, model_type: str, webapp_dir: str, svm_image_size: tuple = (64, 128)) -> tuple[list, list, list]:
    """
    Extract parking spot data from an XML file and process the images according to model type.

    Args:
        xml_path (str): Path to XML file containing parking spot information
        model_type (str): 'svm' or 'dl' to determine processing method
        webapp_dir (str): Base directory of the webapp
        svm_image_size (tuple): Image size for SVM model (default: (64, 128))

    Returns:
        processed_spots: List of processed images
        labels: List of ground truth labels
        contours: List of contour coordinates
    Args:
        xml_path: Path to XML file containing parking spot information
        model_type: 'svm' or 'dl' to determine processing method
        webapp_dir: Base directory of the webapp
        svm_image_size: Image size for SVM model (default: (64, 128))
        
    Returns:
        processed_spots: List of processed images
        labels: List of ground truth labels
        contours: List of contour coordinates
    """
    processed_spots, labels, contours = [], [], []
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(webapp_dir, xml_path)
    image_path = xml_path.replace('.xml', '.jpg')
    if not os.path.exists(image_path):
        return [], [], []
    full_image = cv2.imread(image_path)
    if full_image is None:
        return [], [], []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for space, points in get_valid_spaces(root):
            label = int(space.get('occupied', 0))
            points_np = np.array(points, dtype="float32")
            contours.append(points_np.astype(int))
            labels.append(label)
            
            if model_type == 'svm':
                # SVM path
                x, y, w, h = cv2.boundingRect(points_np.astype(np.int32))
                spot_image = full_image[y:y+h, x:x+w]
                if spot_image.size == 0:
                    continue
                resized_spot = cv2.resize(spot_image, svm_image_size)
                gray_spot = cv2.cvtColor(resized_spot, cv2.COLOR_BGR2GRAY)
                processed_spots.append(gray_spot)
            else:
                # DL path using perspective transformation
                warped_spot = perspective.four_point_transform(full_image, points_np)
                
                if warped_spot.size > 0:
                    spot_image_rgb = cv2.cvtColor(warped_spot, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(spot_image_rgb)
                    processed_spots.append(pil_image)
    except Exception as e:
        print(f"Error processing XML: {e}")
        return [], [], []
    
    return processed_spots, labels, contours


def extract_hog_features(images: list, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG features from images for SVM model.
    
    Args:
        images: List of grayscale images
        orientations: Number of orientation bins for HOG
        pixels_per_cell: Cell size for HOG
        cells_per_block: Block size for HOG
        
    Returns:
        np.array of HOG features
    """
    hog_features = []
    for image in images:
        features = hog(image, 
                       orientations=orientations, 
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, 
                       block_norm='L2-Hys', 
                       visualize=False)
        hog_features.append(features)
    
    return np.array(hog_features)
