#데이터 로딩과 전처리
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 파일 경로 설정
file_path = 'C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movielens.csv'

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, sparse_features):
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    return df

