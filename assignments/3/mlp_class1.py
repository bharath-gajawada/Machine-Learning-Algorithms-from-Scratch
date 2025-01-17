import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import json

import sys
sys.path.append("../..")

import models.MLP.MLPClassifier as MLPclas
from models.knn.knn import Metrics


def split(X, y, val_size=0.2, test_size=0.2):
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

def normalize(X):
    return (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))

def onehot_encode(y):
    unique_classes = np.unique(y)
    one_hot_encoded = np.zeros((y.size, unique_classes.size))
    for i, value in enumerate(y):
        one_hot_encoded[i, np.where(unique_classes == value)] = 1
    
    return one_hot_encoded


best_model = None
best_accuracy = -np.inf
best_params = {}

hyperparameter_table = wandb.Table(columns=["Learning Rate", "Activation", "Optimizer", "Hidden Layers", "Neurons per Layer", "Batch Size", "Epochs", "Early Stopping", "Patience", "Accuracy", "Precision", "Recall", "F1 Score"])

def sweep_agent_manager(X_train, y_train, X_val, y_val):
    
    wandb.init(project="mlp_classifier_hyperparameter_tuning")
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


    accuracy, precision, recall, f1, model = train_and_log_mlp(X_train, y_train, X_val, y_val, config)

    global best_accuracy, best_model, best_params, hyperparameter_table

    hyperparameter_table.add_data(learning_rate, activation, optimizer, hidden_layers, neurons_per_layer, batch_size, epochs, early_stopping, patience, accuracy, precision, recall, f1)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
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
    model = MLPclas.MLPClassifier(learning_rate=config.learning_rate,activation=config.activation,optimizer=config.optimizer,hidden_layers=config.model['hidden_layers'],neurons_per_layer=config.model['neurons_per_layer'],batch_size=config.batch_size,epochs=config.epochs,early_stopping=config.early_stopping,patience=config.patience)

    model.fit(X_train.T, y_train.T)
    y_val_pred = model.predict(X_val.T)

    y_val = np.argmax(y_val, axis=1)

    metrics = Metrics(y_val, y_val_pred)
    acc = metrics.accuracy()
    prec = metrics.precision(method='macro')
    rec = metrics.recall(method='macro')
    f1 = metrics.f1(method='macro')

    wandb.log({
        "accuracy": acc,
        "precision(macro)": prec,
        "recall(macro)": rec,
        "f1_score": f1,
    })

    return acc, prec, rec, f1, model


def train_with_tracking(X, y, model,  batch_size,epochs=100):

        
        input_neurons = X.shape[0]
        output_neurons = y.shape[0]
        
        model.init_params(input_neurons, output_neurons)
        losses = []

        for epoch in range(epochs):
            for i in range(0, X.shape[1], batch_size):
                X_batch = X[:, i:i + batch_size]
                y_batch = y[:, i:i + batch_size]
                A_final, AZ = model.forward_prop(X_batch)
                dW, db = model.back_prop(y_batch, A_final, AZ)
                model.update_weights(dW, db)

            A_final, _ = model.forward_prop(X)
            loss = model.cross_entropy(y, A_final)
            losses.append(loss)
        
        return losses


def main():
    df = pd.read_csv('../../data/external/WineQT.csv')

    # 2.1
    
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

    #                             Mean     Std Dev      Min         Max
    # fixed acidity           8.311111    1.747595  4.60000    15.90000
    # volatile acidity        0.531339    0.179633  0.12000     1.58000
    # citric acid             0.268364    0.196686  0.00000     1.00000
    # residual sugar          2.532152    1.355917  0.90000    15.50000
    # chlorides               0.086933    0.047267  0.01200     0.61100
    # free sulfur dioxide    15.615486   10.250486  1.00000    68.00000
    # total sulfur dioxide   45.914698   32.782130  6.00000   289.00000
    # density                 0.996730    0.001925  0.99007     1.00369
    # pH                      3.311015    0.156664  2.74000     4.01000
    # sulphates               0.657708    0.170399  0.33000     2.00000
    # alcohol                10.442111    1.082196  8.40000    14.90000
    # quality                 5.657043    0.805824  3.00000     8.00000
    # Id                    804.969379  463.997116  0.00000  1597.00000


    plt.figure(figsize=(16, 8))

    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 5, i)

        plt.hist(df[column], bins='auto')

        plt.title(column)
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    # plt.show()
    plt.savefig('figures/distribution(WineQT).png')


    df.fillna(df.mean(), inplace=True)

    X = df.drop(['quality', 'Id'], axis=1).values
    y = df['quality'].values

    X = normalize(X)

    # 2.3

    one_hot_y = onehot_encode(y)
    X_train, y_train, X_val, y_val, X_test, y_test = split(X, one_hot_y)


    sweep_config = {
        'method': 'random',
        'parameters': {
            'learning_rate': {'values': [0.001, 0.01, 0.1, 0.0001] },
            'activation': {'values': ['relu', 'tanh', 'sigmoid', 'linear']},
            'optimizer': {'values': ['sgd', 'batch', 'mini_batch']},
            'model': {
                'values': [
                    {'hidden_layers': 1, 'neurons_per_layer': (64,)},    
                    {'hidden_layers': 1, 'neurons_per_layer': (32,)},    
                    {'hidden_layers': 1, 'neurons_per_layer': (16,)},    
                    {'hidden_layers': 2, 'neurons_per_layer': (64, 32)},
                    {'hidden_layers': 2, 'neurons_per_layer': (32, 16)},
                    {'hidden_layers': 2, 'neurons_per_layer': (64, 16)},
                    {'hidden_layers': 3, 'neurons_per_layer': (128, 64, 32)},
                    {'hidden_layers': 3, 'neurons_per_layer': (64, 32, 16)},
                    {'hidden_layers': 4, 'neurons_per_layer': (128, 64, 32, 16)},
                ]
            },
            'batch_size': {'values': [16, 32, 64, 128]},
            'epochs': {'values': [50, 100, 200]},
            'early_stopping': {'values': [True, False]},
            'patience': {'values': [5, 10]}
        }
    }


    sweep_id = wandb.sweep(sweep=sweep_config, project="mlp_classifier_hyperparameter_tuning")

    wandb.agent(sweep_id, function=lambda: sweep_agent_manager(X_train, y_train, X_val, y_val), count=100)

    wandb.init(project="mlp_classifier_hyperparameter_tuning")
    wandb.log({"Hyperparameter Metrics": hyperparameter_table})

    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best Hyperparameters: {best_params}")

    with open("../../data/interim/best_hyperparams(mlp_classifier).json", 'w') as file:
        json.dump(best_params, file)


    # 2.4
    
    model = MLPclas.MLPClassifier(learning_rate=best_params['learning_rate'],activation=best_params['activation'],optimizer=best_params['optimizer'],hidden_layers=best_params['hidden_layers'],neurons_per_layer=best_params['neurons_per_layer'],batch_size=best_params['batch_size'],epochs=best_params['epochs'],early_stopping=best_params['early_stopping'],patience=best_params['patience'])

    model.fit(X_train.T, y_train.T)
    y_test_pred = model.predict(X_test.T)

    y_test = np.argmax(y_test, axis=1)

    metrics = Metrics(y_test, y_test_pred)

    print(f"Accuracy: {metrics.accuracy()}")
    print(f"Precision: {metrics.precision(method='macro')}")
    print(f"Recall: {metrics.recall(method='macro')}")
    print(f"F1 Score: {metrics.f1(method='macro')}")


    # Accuracy: 0.5964912280701754
    # Precision: 0.27787184352649247
    # Recall: 0.30643660088194896
    # F1 Score: 0.29145600761346446


    
    # 2.5
    activations = ['relu', 'sigmoid', 'tanh', 'linear']
    activation_losses = {}

    for act in activations:
        model = MLPclas.MLPClassifier(learning_rate=best_params['learning_rate'],activation=act,optimizer='mini-batch',hidden_layers=best_params['hidden_layers'],neurons_per_layer=best_params['neurons_per_layer'],batch_size=best_params['batch_size'],epochs=best_params['epochs'],early_stopping=best_params['early_stopping'],patience=best_params['patience'])
        activation_losses[act] = train_with_tracking(X.T, one_hot_y.T, model, best_params['batch_size'], epochs=100)

    plt.figure(figsize=(10, 6))
    for activation in activations:
        plt.plot(activation_losses[activation], label=f'Activation: {activation}')
    plt.title('Loss vs Epochs for Different Activation Functions')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('figures/loss_vs_epochs(activation_functions).png')


    learning_rates = [0.001, 0.01, 0.1, 1.0]
    learning_rate_losses = {}

    for lr in learning_rates:
        model = MLPclas.MLPClassifier(learning_rate=lr,activation=best_params['activation'],optimizer='mini-batch',hidden_layers=best_params['hidden_layers'],neurons_per_layer=best_params['neurons_per_layer'],batch_size=best_params['batch_size'],epochs=best_params['epochs'],early_stopping=best_params['early_stopping'],patience=best_params['patience'])
        learning_rate_losses[lr] = train_with_tracking(X.T, one_hot_y.T, model, best_params['batch_size'], epochs=100)

    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        plt.plot(learning_rate_losses[lr], label=f'Learning Rate: {lr}')
    plt.title('Loss vs Epochs for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('figures/loss_vs_epochs(learning_rates).png')


    batch_sizes = [16, 32, 64, 128]
    batch_size_losses = {}

    for batsz in batch_sizes:
        model = MLPclas.MLPClassifier(learning_rate=best_params['learning_rate'],activation=best_params['activation'],optimizer='mini-batch',hidden_layers=best_params['hidden_layers'],neurons_per_layer=best_params['neurons_per_layer'],batch_size=batsz,epochs=best_params['epochs'],early_stopping=best_params['early_stopping'],patience=best_params['patience'])
        batch_size_losses[batsz] = train_with_tracking(X.T, one_hot_y.T, model, batsz ,epochs=100)

    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        plt.plot(batch_size_losses[batch_size], label=f'Batch Size: {batch_size}')
    plt.title('Loss vs Epochs for Different Batch Sizes (Mini-Batch)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('figures/loss_vs_epochs(batch_sizes).png')


if __name__ == "__main__":
    main()



