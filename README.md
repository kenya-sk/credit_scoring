## Introduction
This repository is credit scoring using data of Kaggle.  
Competition name: Home Credit Default Risk  
URL: https://www.kaggle.com/c/home-credit-default-risk

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
By executing ```src/model/submission.py```, the submission file saved in all/submission.csv.
