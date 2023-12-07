#데이터 로딩과 전처리
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat,get_feature_names

#file_path = 'C:\\Users\\user\\Documents\\GitHub\\pokemonvgg16\\movielens.csv'


class PreProcess:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        return df

    def preprocess_data(self, df, sparse_features=['userId', 'title']):
        for feat in sparse_features:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

        # LabelEncoder 객체 생성 및 저장
        title_encoder = LabelEncoder()
        df['title_encoded'] = title_encoder.fit_transform(df['title'])

        # 'userId'에 대해서도 LabelEncoder 사용 및 저장
        user_encoder = LabelEncoder()
        df['userId_encoded'] = user_encoder.fit_transform(df['userId'])

        # 특성 정의
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                                  for feat in sparse_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        return df, feature_names


