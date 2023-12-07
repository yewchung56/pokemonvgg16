from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names

def build_model(sparse_features, df):
    # 특성 정의
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4) 
                              for feat in sparse_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 모델 생성
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

    return model