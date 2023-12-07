import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat,get_feature_names
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from process import PreProcess
from predict import generate_prediction_data, save_predictions
from train import train_and_evaluate
from model import build_model

preprocess = PreProcess('C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movielens.csv')
df = preprocess.load_data()
df, feature_names = preprocess.preprocess_data(df)
sparse_features = ['userId', 'title'] 
title_encoder = preprocess.get_title_encoder()  # title_encoder 추출
userId_encoder = preprocess.get_userId_encoder()  # title_encoder 추출
model = build_model(sparse_features, df)

# 모델 훈련 및 평가
history, pred_ans = train_and_evaluate(model, df, title_encoder,userId_encoder, 'target')

# 예측 데이터 생성 및 모델을 사용한 예측 수행
prediction_data = generate_prediction_data(df, model,userId_encoder, title_encoder)

# 예측 결과 저장
save_predictions(prediction_data, 'C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movie.csv')







