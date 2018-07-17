## Introduction
This repository is credit scoring using data of Kaggle.  
Competition name: Home Credit Default Risk  
URL: https://www.kaggle.com/c/home-credit-default-risk

## Directory Structure
root/  
&emsp;├ all/  
&emsp;│&emsp;├ application_train.csv  
&emsp;│&emsp;├ application_test.csv  
&emsp;│&emsp;├ bureau.csv  
&emsp;│&emsp;├ credit_card_balance.csv  
&emsp;│&emsp;├ POS_CASH_balance.csv  
&emsp;│&emsp;├ installments_payments.csv  
&emsp;│&emsp;├ previous_application.csv  
&emsp;│&emsp;├ train_X.npy  
&emsp;│&emsp;├ train_target.npy  
&emsp;│&emsp;├ test.npy  
&emsp;│&emsp;├ out_gbm.npy  
&emsp;│&emsp;├ out_mlp.npy  
&emsp;│&emsp;├ submission.csv  
&emsp;│&emsp;└ tensor_model/  
&emsp;│&emsp;&ensp;  
&emsp;└  src/  
&emsp;&emsp;&ensp;├ dataset/  
&emsp;&emsp;&ensp;│&emsp;├ bureau.py  
&emsp;&emsp;&ensp;│&emsp;├ credit_card_balance.py   
&emsp;&emsp;&ensp;│&emsp;├ pos_cash_balance.py   
&emsp;&emsp;&ensp;│&emsp;├ installments_payment.py  
&emsp;&emsp;&ensp;│&emsp;└ make_dataset.py   
&emsp;&emsp;&ensp;│   
&emsp;&emsp;&ensp;└ model/  
&emsp;&emsp;&emsp;&emsp;&emsp;├ model_lightgbm.py   
&emsp;&emsp;&emsp;&emsp;&emsp;├ model_mlp.py  
&emsp;&emsp;&emsp;&emsp;&emsp;└ submission.py  

## Requirement
* Python3
* scikit-learn
* LightGBM (https://github.com/Microsoft/LightGBM)
* Tensor Flow

## Preprocessing & Feature Engineering
Preprocessing and feature engineering for each csv file is done four files in dataset directory.  
* dataset/bureau.py
* dataset/credit_card_balance.py
* dataset/pos_cash_balance.py
* dataset/installments_payment.py

By executing ```dataset/make_dataset.py```, the training data and test data subjected to the preprocessing   
and feature engineering are output.

## Train Model And Prediction
To train lightgbm model and mlp model.  
To set files (train_X.npy, train_target.npy, test.npy) in all directory. There files created by previous step.  
Execut below command.
```
python3 model_lightgbm.py
python3 model_mlp.py
```
Leared model save in all/tensor_model.  
The prediction of each model is saved in all/out_gbm and all/out_mlp.  

## Submission
By executing ```src/model/submission.py```, the submission file saved in all directory.
