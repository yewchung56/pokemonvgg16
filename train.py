from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score


def train_and_evaluate(model, df, feature_names, target):
    # 데이터 분할
    train, test = train_test_split(df, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 모델 훈련 및 평가
    model.compile("adam", "binary_crossentropy", metrics=['AUC'])
    history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=20, verbose=2, validation_split=0.2)

    pred_ans = model.predict(test_model_input, batch_size=256)
    
    target=['target']

    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    return history, pred_ans


