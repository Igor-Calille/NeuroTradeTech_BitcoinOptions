from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import xgboost as xgb

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
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 40],
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
    
    def RandomizedSearchCV_RandomForestRegressor(X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestRegressor(random_state=42)

        param_distributions = {
            'n_estimators': randint(50, 500),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': randint(10, 100),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'bootstrap': [True, False]
        }

        rsearch = RandomizedSearchCV(estimator=model, cv=tscv, param_distributions=param_distributions, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=100, verbose=2, random_state=42)
        rsearch.fit(X, y)
        best_model = rsearch.best_estimator_

        return best_model
    
class using_LinearRidge():
    def __init__(self):
        pass
    
    def normal_split_LinearRidge(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(random_state=42)
        model.fit(X_train, y_train)
        return model

    def GridSearchCV_LinearRidge(X,y):
        tscv = TimeSeriesSplit(n_splits=5)

        model = Ridge(random_state=42)

        param_Search ={
            'alpha': np.logspace(-3, 3, 7) # 10^-3 até 10^3
        }

        gsearch = GridSearchCV(estimator=model, param_grid=param_Search, cv=tscv, scoring='neg_mean_squared_error')

        gsearch.fit(X,y)

        best_model = gsearch.best_estimator_

        return best_model
    
class using_LNN():
    def __init__(self):
        pass

    def LNN_LSTM(X, y):

        # Verifica se a GPU está disponível
        if tf.config.experimental.list_physical_devices('GPU'):
            print("\n----------------Treinando com GPU----------------\n")
        else:
            print("\n----------------Treinando com CPU----------------\n")
            
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Ajusta o formato para (samples, timesteps, features)

        seq_length = X.shape[1]  # Determina dinamicamente o comprimento da sequência
        input_dim = X.shape[2]   # Número de características (dimensão do input)

        model = Sequential()
        model.add(Bidirectional(LSTM(300, activation='tanh', input_shape=(seq_length, input_dim), return_sequences=True, kernel_regularizer='l2')))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(300, activation='tanh', kernel_regularizer='l2')))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        tscv = TimeSeriesSplit(n_splits=5)

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stopping])

        return model
