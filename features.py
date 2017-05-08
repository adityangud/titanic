from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Sex(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['sex'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        sex_value = lambda l1: 1 if l1 == "male" else 2
        sex = [sex_value(d) for d in documents]
        return np.array([sex]).T

class Embarked(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['embarked'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        embarked = []
        for d in documents:
            if d == "C":
                value = 1
            elif d == "S":
                value = 2
            else:
                value = 3
            embarked.append(value)

        return np.array([embarked]).T

class Pclass(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['pclass'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        pclass = [d for d in documents]
        return np.array([pclass]).T

class SibSp(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['sibsp'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        sibsp = [d for d in documents]
        return np.array([sibsp]).T

class Parch(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['parch'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        parch = [d for d in documents]
        return np.array([parch]).T

class Fare(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['fare'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        fare = [d for d in documents]
        return np.array([fare]).T

class Cabin(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['cabin'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        A = ord('A')
        cabin  = []
        for d in documents:
            if d == "":
                value = 0
            else:
                for c in d:
                    if c.isalpha():
                        value = ord(c) - A + 1
                        break
            cabin.append(value)
        return np.array([cabin]).T

