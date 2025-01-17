import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../..")

import models.linear_regression.linear_regression as lr

def split(X, y, test_size=0.1, val_size=0.1):
        np.random.seed(42)
        ind = np.random.permutation(len(X))
        
        test_size = int(len(X) * test_size)
        val_size = int(len(X) * val_size)
        
        test_ind = ind[:test_size]
        val_ind = ind[test_size:test_size + val_size]
        train_ind = ind[test_size + val_size:]

        X_train, X_val, X_test = X[train_ind], X[val_ind], X[test_ind]
        y_train, y_val, y_test = y[train_ind], y[val_ind], y[test_ind]

        return X_train, y_train, X_val, y_val, X_test, y_test

def main():
        df = pd.read_csv('../../data/external/linreg.csv')
        df.dropna(inplace=True)

        X = df['x'].to_numpy()
        y = df['y'].to_numpy()

        X_train, y_train, X_val, y_val, X_test, y_test = split(X, y)

        
        #3.1.1
        model = lr.LinearRegression(k=1)
        model.fit(X_train, y_train)
        metrics = model.validate(X_val, y_val)
        print(f'Degree 1 Model Metrics: {metrics}')

        y_pred = model.predict(X_test)
        plt.scatter(X, y, color='red')
        plt.plot(X_test, y_pred, color='blue')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Degree 1 Model')
        plt.show()

        # Degree 1 Model Metrics: {'mse': 0.655974153027033, 'variance': 0.03214969691222917, 'std_dev': 0.1793033655909146}

        #3.1.2
        best_k = None
        best_metrics = None
        for i in range(2, 21):
            model = lr.LinearRegression(k=i)
            model.fit(X_train, y_train)
            metrics = model.validate(X_val, y_val)
            print(f'Degree {i} Model Metrics: {metrics}')
            
            if best_metrics is None or metrics['mse'] < best_metrics['mse']:
                best_k = i
                best_metrics = metrics

        print(f'Best Degree: {best_k}')
        print(f'Best Model Metrics: {best_metrics}')

        # Degree 2 Model Metrics: {'mse': 0.63797389616287, 'variance': 0.054929495560163376, 'std_dev': 0.23437042381700676}
        # Degree 3 Model Metrics: {'mse': 0.2106729285536555, 'variance': 0.3328614462304857, 'std_dev': 0.5769414582351364}
        # Degree 4 Model Metrics: {'mse': 0.22297161787239758, 'variance': 0.34157231356835493, 'std_dev': 0.5844418821134869}
        # Degree 5 Model Metrics: {'mse': 0.22939851738581404, 'variance': 0.3212547373480917, 'std_dev': 0.566793381531658}
        # Degree 6 Model Metrics: {'mse': 0.2695618339972286, 'variance': 0.31454500092146104, 'std_dev': 0.5608431161398534}
        # Degree 7 Model Metrics: {'mse': 0.1539030438868226, 'variance': 0.3942729320294231, 'std_dev': 0.6279115638602486}
        # Degree 8 Model Metrics: {'mse': 0.14206615355920219, 'variance': 0.41340125674553957, 'std_dev': 0.6429628735358983}
        # Degree 9 Model Metrics: {'mse': 0.12807902650611833, 'variance': 0.49310408572207465, 'std_dev': 0.7022137037413003}
        # Degree 10 Model Metrics: {'mse': 0.1425551220284516, 'variance': 0.5294743662422159, 'std_dev': 0.7276498926284645}
        # Degree 11 Model Metrics: {'mse': 0.13971822388925928, 'variance': 0.4760433086858013, 'std_dev': 0.6899589181145507}
        # Degree 12 Model Metrics: {'mse': 0.13881013751731325, 'variance': 0.46394917819460985, 'std_dev': 0.6811381491258656}
        # Degree 13 Model Metrics: {'mse': 0.23908925444968107, 'variance': 0.43264388666456705, 'std_dev': 0.6577567078065925}
        # Degree 14 Model Metrics: {'mse': 0.15437313918992998, 'variance': 0.4372801936508875, 'std_dev': 0.6612716489090452}
        # Degree 15 Model Metrics: {'mse': 0.1644515196799985, 'variance': 0.42997453336249974, 'std_dev': 0.6557244340136333}
        # Degree 16 Model Metrics: {'mse': 0.16088534964940893, 'variance': 0.45529026855205224, 'std_dev': 0.6747520052226983}
        # Degree 17 Model Metrics: {'mse': 0.16447473530101725, 'variance': 0.46604662542794684, 'std_dev': 0.6826760765018406}
        # Degree 18 Model Metrics: {'mse': 0.19155051430425094, 'variance': 0.4908871633131485, 'std_dev': 0.7006334015111958}
        # Degree 19 Model Metrics: {'mse': 0.1697988047617262, 'variance': 0.44800729065647404, 'std_dev': 0.669333467455852}
        # Degree 20 Model Metrics: {'mse': 0.21410265229121536, 'variance': 0.4979986293968971, 'std_dev': 0.7056901794675176}
        # Best Degree: 9
        # Best Model Metrics: {'mse': 0.12807902650611833, 'variance': 0.49310408572207465, 'std_dev': 0.7022137037413003}


if __name__ == '__main__':
        main()