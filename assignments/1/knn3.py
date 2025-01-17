import pandas as pd
import numpy as np

import sys
sys.path.append("../..")

import models.knn.knn as knn

def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['explicit'] = df['explicit'].astype(int)
    df['explicit']
    df.drop(['track_id','artists','album_name','track_name'], axis=1, inplace=True)
    df['track_genre'] = pd.Categorical(df['track_genre']).codes

    df.drop('Unnamed: 0', inplace=True, axis=1)
    return df

def main():
    best_hyper_params = np.load('../../data/interim/best_hyper_params(knn).npy')

    train = pd.read_csv('../../data/external/spotify-2/train.csv')
    train = clean_data(train)

    X_train = train.drop('track_genre', axis=1).to_numpy()
    y_train = train['track_genre'].to_numpy()
    
    test = pd.read_csv('../../data/external/spotify-2/test.csv')
    test = clean_data(test)

    X_test = test.drop('track_genre', axis=1).to_numpy()
    y_test = test['track_genre'].to_numpy()

    validation = pd.read_csv('../../data/external/spotify-2/validate.csv')
    validation = clean_data(validation)

    X_val = validation.drop('track_genre', axis=1).to_numpy()
    y_val = validation['track_genre'].to_numpy()

    #2.6

    best_hyper_params = np.load('../../data/interim/best_hyper_params(knn).npy')
    _,best_k,best_distance_type = best_hyper_params[0]
    best_k = best_k.astype(int)
    best_distance_type = best_distance_type.tolist()
    model = knn.KNN(k=best_k, distance_type=best_distance_type)
    model.fit(X_train, y_train)
    metrics = model.validate(X_val, y_val)
    print(f'Best Model Metrics: {metrics}')

    # Best Model Metrics: {'accuracy': 0.03456140350877193, 'precision micro': 0.03456140350877193, 'precision macro': 0.028222437940019593, 'recall micro': 0.03456140350877193, 'recall macro': 0.035026754750866404, 'f1 micro': 0.03456140350877193, 'f1 macro': 0.03125859382989592}


if __name__ == "__main__":
    main()
