# src/model.py

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.data_preprocessing import preprocess_data

def create_pipelines():
    preprocessor = preprocess_data()
    pipelines = {
        'linear_regression': Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())]),
        'decision_tree': Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor())]),
        'random_forest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor())])
    }
    return pipelines
