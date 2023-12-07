import os
import numpy as np
from featEx import FeatureExtractor
from PIL import Image
import matplotlib.pyplot as plt

fe = FeatureExtractor()
features = []
img_paths = []
 # 이미지가 있는 폴더 경로
directory ="C:\\Users\\user\\Downloads\\pokemon\\images\\images\\"
# # 폴더 내 모든 파일 이름을 가져옵니다.
img_files = [f for f in os.listdir(directory) if f.endswith('.png')]
for img_name in img_files:
    try:
        image_path = os.path.join(directory, img_name)
        img_paths.append(image_path)
        # Extract Features
        feature = fe.extract(img=Image.open(image_path))
        features.append(feature)
        # Save the Numpy array (.npy) on designated path
        feature_path = "C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\features\\" + (img_name)[0] + ".npy"
        np.save(feature_path, feature)
    except Exception as e:
        print('예외가 발생했습니다.', e)



# Target 이미지
#포켓몬이미지를 변수로 만들어서
img = Image.open("C:\\Users\\user\\Downloads\\pokemon\\images\\images\\ampharos.png")
query = fe.extract(img)
# 유사도 계산
dists = np.linalg.norm(features - query, axis=1)
ids = np.argsort(dists)[:30]
scores = [(dists[id], img_paths[id]) for id in ids]
axes=[]
fig=plt.figure(figsize=(8,8))
for a in range(5*6):
    score = scores[a]
    axes.append(fig.add_subplot(5, 6, a+1))
    subplot_title=str(score[0])
    axes[-1].set_title(subplot_title)
    plt.axis('off')
    plt.imshow(Image.open(score[1]))
fig.tight_layout()
plt.show()
#유클리드거리: 숫자가 작을수록 가까운 것.