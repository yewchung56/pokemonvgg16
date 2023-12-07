import pandas as pd
from sklearn.preprocessing import LabelEncoder

def generate_prediction_data(df, model, title_encoder):
    users = df['userId'].unique()
    movies = df['title_encoded'].unique()
    prediction_data = pd.DataFrame([(user, movie) for user in users for movie in movies], columns=['userId', 'title_encoded'])

    # 모델 입력 형식화
    prediction_model_input = {'userId': prediction_data['userId'], 'title': prediction_data['title_encoded']}

    # 모델을 사용한 예측 수행
    predictions = model.predict(prediction_model_input, batch_size=256)

    # 결과 저장 전, 'title_encoded' 열을 원래 영화 제목으로 역변환
    prediction_data['title'] = title_encoder.inverse_transform(prediction_data['title_encoded'])

    
    prediction_data['probability'] = predictions
    return prediction_data

def save_predictions(prediction_data, file_path):
    prediction_data[['userId', 'title', 'probability']].to_csv('C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movie.csv', index=False)


