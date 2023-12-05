from PIL import Image
import os
import numpy as np

def load_images_from_directory(directory):
    img_paths = []
    img_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    for img_name in img_files:
        image_path = os.path.join(directory, img_name)
        img_paths.append(image_path)
    return img_paths

def save_feature(feature, feature_path):
    np.save(feature_path, feature)