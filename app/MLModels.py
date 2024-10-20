from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
import numpy as np


class using_RandomForestRegressor():
    def __init__(self):
        pass

    def normal_split_RandomForestRegressor(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def GridSearchCV_RandomForestRegressor(X,y):
        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestRegressor(random_state=42)

        
        param_search = {
            'n_estimators': [50, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        gsearch.fit(X, y)
        best_model = gsearch.best_estimator_

       
        tree_predictions = np.array([tree.predict(X) for tree in best_model.estimators_])

       
        mean_predictions = np.mean(tree_predictions, axis=0)

       
        std_predictions = np.std(tree_predictions, axis=0)

        
        return best_model, mean_predictions, std_predictions
    
    