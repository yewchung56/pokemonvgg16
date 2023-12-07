import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat,get_feature_names
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from process import PreProcess
from train import train_and_evaluate


preprocess = PreProcess('C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movielens.csv')
df = preprocess.load_data()
df, feature_names = preprocess.preprocess_data(df)




# 데이터셋 생성
users = df['userId'].unique()
movies = df['title_encoded'].unique()
prediction_data = pd.DataFrame([(user, movie) for user in users for movie in movies], columns=['userId', 'title_encoded'])

# 모델 입력 형식화
prediction_model_input = {'userId': prediction_data['userId'], 'title': prediction_data['title_encoded']}

# 모델을 사용한 예측 수행
predictions = model.predict(prediction_model_input, batch_size=256)

# 결과 저장 전, 'title_encoded' 열을 원래 영화 제목으로 역변환
prediction_data['title'] = lbe.inverse_transform(prediction_data['title_encoded'])



# 결과 저장
prediction_data['probability'] = predictions
prediction_data[['userId', 'title', 'probability']].to_csv('C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movie.csv', index=False)

