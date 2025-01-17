import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import json

import sys
sys.path.append("../..")

import models.MLP.MLPRegression as MLPreg

def split_data(X, y, val_size=0.2, test_size=0.2):
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_val = int(n * val_size)
    n_test = int(n * test_size)

    val_indices = indices[:n_val]
    test_indices = indices[n_val:n_val + n_test]
    train_indices = indices[n_val + n_test:]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    X_train = X[train_indices]
    y_train = y[train_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test

best_model = None
best_mse = np.inf
best_params = {}

hyperparameter_table = wandb.Table(columns=["Learning Rate", "Activation", "Optimizer", "Hidden Layers", "Neurons per Layer", "Batch Size", "Epochs", "Early Stopping", "Patience", "MSE", "RMSE", "R^2"])

def sweep_agent_manager(X_train, y_train, X_val, y_val):
    
    wandb.init(project="mlp_regression_hyperparameter_tuning")
    config = wandb.config

    learning_rate = config.learning_rate
    activation = config.activation
    optimizer = config.optimizer
    hidden_layers = config.model['hidden_layers']  
    neurons_per_layer = config.model['neurons_per_layer']
    batch_size = config.batch_size
    epochs = config.epochs
    early_stopping = config.early_stopping
    patience = config.patience
    
    run_name = f"lr = {learning_rate} | act = {activation} | opt = {optimizer} | layers = {hidden_layers} | neurons = {neurons_per_layer} | bat_sz = {batch_size} | ep = {epochs} | ear = {early_stopping} | pat = {patience}"

    wandb.run.name = run_name


    mse, rmse, r_squared, model = train_and_log_mlp(X_train, y_train, X_val, y_val, config)

    global best_mse, best_model, best_params, hyperparameter_table

    hyperparameter_table.add_data(learning_rate, activation, optimizer, hidden_layers, neurons_per_layer, batch_size, epochs, early_stopping, patience, mse, rmse, r_squared)
    
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_params = {
            "learning_rate": learning_rate,
            "activation": activation,
            "optimizer": optimizer,
            "hidden_layers": hidden_layers,
            "neurons_per_layer": neurons_per_layer,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": early_stopping,
            "patience": patience
        }


def train_and_log_mlp(X_train, y_train, X_val, y_val, config):
    model = MLPreg.MLPRegressor(learning_rate=config.learning_rate,activation=config.activation,optimizer=config.optimizer,hidden_layers=config.model['hidden_layers'],neurons_per_layer=config.model['neurons_per_layer'],batch_size=config.batch_size,epochs=config.epochs,early_stopping=config.early_stopping,patience=config.patience)

    model.fit(X_train.T, y_train.T)
    y_val_pred = model.predict(X_val.T)


    mse = model.mse(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_val - y_val_pred) ** 2)
    mean = np.mean(y_val)
    ss_tot = np.sum((y_val - mean) ** 2)

    r_squared = 1 - (ss_res / ss_tot)


    wandb.log({
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r_squared
    })

    return mse, rmse, r_squared, model


def normalize(X):
    return (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))

def main():
    df = pd.read_csv('../../data/external/HousingData.csv')

    # 3.1

    mean = df.mean()
    std = df.std()
    min = df.min()
    max = df.max()

    val = pd.DataFrame({
        'Mean': mean,
        'Std Dev': std,
        'Min': min,
        'Max': max
    })

    print(val)

    #                Mean     Std Dev        Min       Max
    # CRIM       3.611874    8.720192    0.00632   88.9762
    # ZN        11.211934   23.388876    0.00000  100.0000
    # INDUS     11.083992    6.835896    0.46000   27.7400
    # CHAS       0.069959    0.255340    0.00000    1.0000
    # NOX        0.554695    0.115878    0.38500    0.8710
    # RM         6.284634    0.702617    3.56100    8.7800
    # AGE       68.518519   27.999513    2.90000  100.0000
    # DIS        3.795043    2.105710    1.12960   12.1265
    # RAD        9.549407    8.707259    1.00000   24.0000
    # TAX      408.237154  168.537116  187.00000  711.0000
    # PTRATIO   18.455534    2.164946   12.60000   22.0000
    # B        356.674032   91.294864    0.32000  396.9000
    # LSTAT     12.715432    7.155871    1.73000   37.9700
    # MEDV      22.532806    9.197104    5.00000   50.0000

    plt.figure(figsize=(16, 8))

    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 5, i)

        plt.hist(df[column], bins='auto')

        plt.title(column)
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    # plt.show()
    plt.savefig('figures/distribution(HousingData).png')

    df.fillna(df.mean(), inplace=True)

    normalized_df = normalize(df)
    X = normalized_df.drop('MEDV', axis=1).values
    y = normalized_df['MEDV'].values
    y = y.reshape(-1, 1)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)


    # 3.3
    sweep_config = {
        'method': 'random',
        'parameters': {
            'learning_rate': {'values': [0.001, 0.01, 0.0001, 0.00001] },
            'activation': {'values': ['relu', 'tanh', 'sigmoid', 'linear']},
            'optimizer': {'values': ['sgd', 'batch', 'mini_batch']},
            'model': {
                'values': [   
                    {'hidden_layers': 1, 'neurons_per_layer': (32,)},    
                    {'hidden_layers': 1, 'neurons_per_layer': (16,)},    
                    {'hidden_layers': 1, 'neurons_per_layer': (2,)},    
                    {'hidden_layers': 1, 'neurons_per_layer': (4,)},    
                    {'hidden_layers': 1, 'neurons_per_layer': (8,)},    
                    {'hidden_layers': 2, 'neurons_per_layer': (32, 16)},
                    {'hidden_layers': 2, 'neurons_per_layer': (16, 8)},
                    {'hidden_layers': 2, 'neurons_per_layer': (8, 4)},
                    {'hidden_layers': 2, 'neurons_per_layer': (4, 2)},
                    {'hidden_layers': 3, 'neurons_per_layer': (32, 16, 8)},
                    {'hidden_layers': 3, 'neurons_per_layer': (16, 16, 16)},
                ]
            },
            'batch_size': {'values': [16, 32, 64, 128]},
            'epochs': {'values': [100, 500, 1000]},
            'early_stopping': {'values': [True]},
            'patience': {'values': [10, 20]}
        }
    }


    sweep_id = wandb.sweep(sweep=sweep_config, project="mlp_regression_hyperparameter_tuning")

    wandb.agent(sweep_id, function=lambda: sweep_agent_manager(X_train, y_train, X_val, y_val), count=100)

    wandb.init(project="mlp_regression_hyperparameter_tuning")
    wandb.log({"Hyperparameter Metrics": hyperparameter_table})

    print(f"Best MSE: {best_mse}")
    print(f"Best Hyperparameters: {best_params}")

    # 3.4

    best_model = MLPreg.MLPRegressor(learning_rate=best_params['learning_rate'],activation=best_params['activation'],optimizer=best_params['optimizer'],hidden_layers=best_params['hidden_layers'],neurons_per_layer=best_params['neurons_per_layer'],batch_size=best_params['batch_size'],epochs=best_params['epochs'],early_stopping=best_params['early_stopping'],patience=best_params['patience'])

    best_model.fit(X_train.T, y_train.T)
    y_test_pred = best_model.predict(X_test.T)

    mse = best_model.mse(y_test, y_test_pred)
    mae = np.mean(np.abs(y_test - y_test_pred))

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    # MSE: 0.032914313325317746
    # MAE: 0.13696050554668926


if __name__ == "__main__":
    main()