#데이터 로딩과 전처리
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat,get_feature_names

#file_path = 'C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movielens.csv'


class PreProcess:
    def __init__(self, file_path):
        self.file_path = file_path
        self.title_encoder = LabelEncoder()  # title_encoder 속성 추가

    def load_data(self):
        df = pd.read_csv(self.file_path)
        return df

    def preprocess_data(self, df, sparse_features=['userId', 'title']):
        for feat in sparse_features:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

        # LabelEncoder 객체 생성 및 저장
        self.title_encoder.fit(df['title'])  # title_encoder 학습
        df['title_encoded'] = self.title_encoder.transform(df['title'])
        # 'userId'에 대해서도 LabelEncoder 사용 및 저장
        user_encoder = LabelEncoder()
        df['userId_encoded'] = user_encoder.fit_transform(df['userId'])

        return df, sparse_features
    def get_title_encoder(self):
        return self.title_encoder  # title_encoder 반환 메소드

