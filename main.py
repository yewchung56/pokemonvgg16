from model.feature_extractor import FeatureExtractor
from utils.image_utils import load_images_from_folder
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로딩 및 특징 추출
fe = FeatureExtractor()
features = []
img_paths = []
directory = "C:\Users\user\Downloads\pokemon\images\images"
img_files = [f for f in os.listdir(directory) if f.endswith('.png')]

for img_name in img_files:
    try:
        image_path = os.path.join(directory, img_name)
        img_paths.append(image_path)
        # Extract Features
        feature = fe.extract(img=Image.open(image_path))
        features.append(feature)
        # Save the Numpy array (.npy) on designated path
        feature_path = "C:\Users\user\Documents\GitHub\pokemonvgg16\\features" ,(os.path.splitext(img_name)[0] + ".npy")
        np.save(feature_path, feature)
    except Exception as e:
        print('예외가 발생했습니다.', e)


