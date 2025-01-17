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
        df = pd.read_csv('../../data/external/regularisation.csv')
        df.dropna(inplace=True)

        X = df['x'].to_numpy()
        y = df['y'].to_numpy()

        X_train, y_train, X_val, y_val, X_test, y_test = split(X, y)
        plt.scatter(X_train, y_train)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Train Data')
        plt.show()
        
    
        #3.2.1
        fig, axs = plt.subplots(5, 4, figsize=(14, 10))
        for i in range(1, 21):
            model = lr.LinearRegression(k=i)
            model.fit(X_train, y_train)
            metrics = model.validate(X_val, y_val)
            y_pred = model.predict(X_test)
            subplt = axs[(i-1)//4][(i-1)%4]
            subplt.scatter(X_train, y_train, color='blue', label='Train Data')
            subplt.scatter(X_test, y_pred, color='red', label='Test Data')
            subplt.set_title(f'Degree {i} Polynomial Regression')
            subplt.set_xlabel('X')
            subplt.set_ylabel('y')
            subplt.legend(fontsize='small')
            print(f'Degree {i} Model Metrics: {metrics}')

        plt.tight_layout()
        plt.savefig('figures/linear_regression(without_regularisation).png')
        # plt.show()

    # Degree 1 Model Metrics: {'mse': 0.2692962982854516, 'variance': 0.01141473418272789, 'std_dev': 0.1068397593722856}
    # Degree 2 Model Metrics: {'mse': 0.09718058600320319, 'variance': 0.06235430185789802, 'std_dev': 0.2497084336939744}
    # Degree 3 Model Metrics: {'mse': 0.28632149699751874, 'variance': 0.12000117774829486, 'std_dev': 0.34641186144284214}
    # Degree 4 Model Metrics: {'mse': 0.29281119235068054, 'variance': 0.11537131209988186, 'std_dev': 0.3396635277740044}
    # Degree 5 Model Metrics: {'mse': 0.20097611082370756, 'variance': 0.06994751455684659, 'std_dev': 0.2644759243425507}
    # Degree 6 Model Metrics: {'mse': 0.08009005937458678, 'variance': 0.27542327652964194, 'std_dev': 0.5248078472447244}
    # Degree 7 Model Metrics: {'mse': 0.14532410909059443, 'variance': 0.2836431288113499, 'std_dev': 0.5325815701010972}
    # Degree 8 Model Metrics: {'mse': 0.15280417665921814, 'variance': 0.23819410711540626, 'std_dev': 0.4880513365573403}
    # Degree 9 Model Metrics: {'mse': 0.18538571039628293, 'variance': 0.25342363567979, 'std_dev': 0.5034119939768916}
    # Degree 10 Model Metrics: {'mse': 0.1869824370246372, 'variance': 0.2346092904566793, 'std_dev': 0.48436483197759034}
    # Degree 11 Model Metrics: {'mse': 0.13030230712217364, 'variance': 0.21627294264964242, 'std_dev': 0.4650515483789323}
    # Degree 12 Model Metrics: {'mse': 0.12585750421978423, 'variance': 0.24496454932056805, 'std_dev': 0.49493893494103697}
    # Degree 13 Model Metrics: {'mse': 0.11444345823060574, 'variance': 0.2469827362696776, 'std_dev': 0.4969735770337067}
    # Degree 14 Model Metrics: {'mse': 0.07027062612150356, 'variance': 0.16070340276685224, 'std_dev': 0.40087828921862584}
    # Degree 15 Model Metrics: {'mse': 0.07942869008201489, 'variance': 0.15608393257353229, 'std_dev': 0.3950745911515094}
    # Degree 16 Model Metrics: {'mse': 0.0583048828706531, 'variance': 0.15147385045063938, 'std_dev': 0.3891964162869943}
    # Degree 17 Model Metrics: {'mse': 0.05431264958165958, 'variance': 0.1599409968521047, 'std_dev': 0.3999262392643232}
    # Degree 18 Model Metrics: {'mse': 0.047651341014223934, 'variance': 0.18805117185001466, 'std_dev': 0.4336486732944247}
    # Degree 19 Model Metrics: {'mse': 0.09112653496121384, 'variance': 0.11266291307363076, 'std_dev': 0.3356529652388472}
    # Degree 20 Model Metrics: {'mse': 0.1979435554843339, 'variance': 0.11025129100717979, 'std_dev': 0.3320410983706381}


        for type in ['L1', 'L2']:
            for lambda_val in [0.01, 0.1, 1, 10]:
                fig, axs = plt.subplots(5, 4, figsize=(14, 10))
                for i in range(1, 21):
                    model = lr.LinearRegression(k=i, reg_type=type, reg_lambda=lambda_val)
                    model.fit(X_train, y_train)
                    metrics = model.validate(X_val, y_val)
                    y_pred = model.predict(X_test)
                    subplt = axs[(i-1)//4][(i-1)%4]
                    subplt.scatter(X_train, y_train, color='blue', label='Train Data')
                    subplt.scatter(X_test, y_pred, color='red', label='Test Data')
                    subplt.set_title(f'Degree {i} {type} Regularisation Lambda {lambda_val}')
                    subplt.set_xlabel('X')
                    subplt.set_ylabel('y')
                    subplt.legend(fontsize='small')
                    # print(f'Degree {i} {type} Regularisation Lambda {lambda_val} Model Metrics: {metrics}')

                plt.tight_layout()
                plt.savefig(f'figures/linear_regression({type}_regularisation({lambda_val})).png')
                # plt.show()

if __name__ == '__main__':
        main()