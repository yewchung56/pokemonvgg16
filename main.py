import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat,get_feature_names
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score

# 파일 경로 설정
file_path = 'C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movielens.csv'

# CSV 파일 로드
df = pd.read_csv(file_path)

# 데이터의 기본 정보 확인
#print(df.info())

#전처리 (범주형: title,, genres, tag)
sparse_features = ['userId', 'title']
for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])


# LabelEncoder 객체 생성 및 저장
title_encoder = LabelEncoder()
df['title_encoded'] = title_encoder.fit_transform(df['title'])

# 'userId'에 대해서도 LabelEncoder 사용 및 저장
user_encoder = LabelEncoder()
df['userId_encoded'] = user_encoder.fit_transform(df['userId'])

#print(df['title_encoded'])

# 특성 정의
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                          for feat in sparse_features]
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 데이터 분할
train, test = train_test_split(df, test_size=0.2, random_state=2020)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 모델 정의 및 컴파일
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['AUC'])

# 모델 훈련
history = model.fit(train_model_input, train['target'].values,
                    batch_size=256, epochs=20, verbose=2, validation_split=0.2)

# 예측 및 평가
pred_ans = model.predict(test_model_input, batch_size=256)

target=['target']

print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))



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

