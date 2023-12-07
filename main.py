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


