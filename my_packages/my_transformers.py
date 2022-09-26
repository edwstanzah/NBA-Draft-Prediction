from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


######################## CUSTOM TRANSFORMER FOR PREPROCESSING ##################
class CustomTransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        ...


    def fit(self, X, y=None):
        return self      
    

    def transform(self, X):
        X = X.copy()
        temp_cols = X.columns.to_list()
        X = add_positions(X)
        
        return X  


def add_positions(data):
    data[['primary_position', 'secondary_position']] = data['position'].str.split('-', expand=True)

    data = data.drop('position', axis=1)
    return data


######################## FEATURE SELECTOR #####################################
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):


    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k


    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

        
    def transform(self, X):
        return X[:, self.feature_indices_]