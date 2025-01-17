# Assignment 2 Report

## 2.3
- [metrics table](https://wandb.ai/bharathgajawada2004-iiit-hyderabad/mlp_classifier_hyperparameter_tuning/reports/Weave-Hyperparameter-Metrics-24-10-12-16-25-22---Vmlldzo5NjkyNzk5?accessToken=7gobleb0lnbt3egq80a2iun8k5nl2kmwx457ixwh13xm4crfra3yqkm2bwb2kq6h)

- [accuracy plot](https://api.wandb.ai/links/bharathgajawada2004-iiit-hyderabad/2984b2cg)

- [precession plot](https://api.wandb.ai/links/bharathgajawada2004-iiit-hyderabad/q9yao939)

- [recall plot](https://wandb.ai/bharathgajawada2004-iiit-hyderabad/mlp_classifier_hyperparameter_tuning/reports/recall-macro-24-10-12-16-27-56---Vmlldzo5NjkyODEx?accessToken=eqbxpv78lmonnm6i9pbamir2tcqs0w9wlaqvw9c2tdlpzm4cngyw2ovzv27yoq42)

- [f1_score plot](https://wandb.ai/bharathgajawada2004-iiit-hyderabad/mlp_classifier_hyperparameter_tuning/reports/f1_score-24-10-12-16-28-31---Vmlldzo5NjkyODE0?accessToken=v106r4ai66krecs7f5mgelcq59p4pu3as65yo9y6uve7itjbyrkrm15qbjfitjvp)
    


## 2.5

- A high learning rate can cause the loss to jump around a lot, making it very unstable, possibly preventing the model from learning. While a low learning rate leads to a slow and steady drop in loss but may take a long time to finish training.

- Small batch sizes can create noisy updates, making the training process bit unstable but sometimes quicker. Large batch sizes provide smoother updates, leading to a steadier decrease in loss, but they can slow down how fast the model learns overall.

- ReLU helps the model learn quickly but can sometimes cause some neurons to stop working. Sigmoid is slow and can get stuck, while tanh works better than sigmoid but still has some issues in vanishing gradients.



## 2.7

- Best hyperparameters(based on accuracy): Learning_rate = 0.1,Activation = relu,optimizater = sgd ,hidden layer = 2,neurons = 16,32, Batch size =  16,Epochs =  100,early stopping = True,patience = 10

- least hyperparameters(based on accuracy): Learning_rate = 0.1,Activation = linear,optimizater = sgd ,hidden layer = 4,neurons = 128, 64, 32, 16, Batch size =  32,Epochs =  100,early stopping = True,patience = 10

- 2nd model has least accuracy beacuse of more layers and neurons.

## 3.3

- [metrics table](https://wandb.ai/bharathgajawada2004-iiit-hyderabad/mlp_regression_hyperparameter_tuning/reports/Weave-Hyperparameter-Metrics-24-10-12-16-35-10---Vmlldzo5NjkyODU1?accessToken=3bwkuurtecan61v31xg5aq5r2xzq7hrs5bpciihkk7f3wbitm0ev5vo5q9opma9g)

- [R^2 plot](https://wandb.ai/bharathgajawada2004-iiit-hyderabad/mlp_regression_hyperparameter_tuning/reports/R-2-24-10-12-16-35-42---Vmlldzo5NjkyODYy?accessToken=20iecd6dejxt0zb6t6was8lukske3w70prvqyn2u7ig2ncu9yoa0vfecp6q1f6g3)

- [RMSE plot](https://wandb.ai/bharathgajawada2004-iiit-hyderabad/mlp_regression_hyperparameter_tuning/reports/RMSE-24-10-12-16-36-11---Vmlldzo5NjkyODY4?accessToken=5aaxhzqt3m7f1maplb4bdtm6vbjmrsjaiyvo9nrmxi1ianosu37unvx9233al64n)

- [MSE plot](https://wandb.ai/bharathgajawada2004-iiit-hyderabad/mlp_regression_hyperparameter_tuning/reports/MSE-24-10-12-16-36-35---Vmlldzo5NjkyODc1?accessToken=ebkssrfpedioxzj8apbxpsk84ja4bqfoipjjlev9bb8ignxsdcdhmi7y7tiyoa9a)

## 3.5
- MSE starts with higher values and exhibits a smoother, slower convergence. In contrast, BCE begins lower, converges more quickly, and shows abrupt changes near the decision boundary, stabilizing at lower values, which reflect better performance in binary classification tasks.

## 3.6

- Best hyperparameters(based on mse): Learning_rate = 0.001,Activation = relu,optimizater = sgd ,hidden layer = 1,neurons = 2, Batch size =  64,Epochs =  500,early stopping = True,patience = 10

- least hyperparameters(based on mse): Learning_rate = 0.00001,Activation = tanh,optimizater = batch ,hidden layer = 1,neurons = 16, Batch size =  128,Epochs =  100,early stopping = True,patience = 10

- 2nd model has least mse because of low learning rate, and less number of epoch, models with similar learning and more epochs performed well



## 4.3

- metrics obtained through PCA, AutoEncoder has are almost similar, because reduced data is 2 dimensional ( obtained from assignment-2), so most of the data is lost, and metric values are low.



