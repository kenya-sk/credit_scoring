#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,MinMaxScaler, Imputer
from sklearn.model_selection import train_test_split

from bureau import merge_bureau
from pos_cash_balance import feature_pos
from credit_card_balance import feature_credit_balance
from installments_payment import feature_installments

def merge_application(application_train_path, application_test_path):
    # Train
    application_train_df = pd.read_csv(application_train_path)
    print("train data shape: {}".format(application_train_df.shape))
    # Test
    application_test_df = pd.read_csv(application_test_path)
    print("test data shape: {}".format(application_test_df.shape))

    # merge bureau data
    bureau_df = pd.read_csv("../../all/bureau.csv")
    bureau_balance_df = pd.read_csv("../../all/bureau_balance.csv")
    merge_bureau_df = merge_bureau(bureau_df, bureau_balance_df)
    application_train_df = application_train_df.merge(merge_bureau_df, on=["SK_ID_CURR"], how="left")
    application_test_df = application_test_df.merge(merge_bureau_df, on=["SK_ID_CURR"], how="left")
    print("train shape (mergerd bureau): {}".format(application_train_df.shape))
    print("test shape (merged bureau): {}\n".format(application_test_df.shape))

    # merge POS data
    pos_df = pd.read_csv("../../all/POS_CASH_balance.csv")
    latest_pos_df = feature_pos(pos_df)
    application_train_df = application_train_df.merge(latest_pos_df, on=["SK_ID_CURR"], how="left")
    application_test_df = application_test_df.merge(latest_pos_df, on=["SK_ID_CURR"], how="left")
    print("train shape (merged POS): {}".format(application_train_df.shape))
    print("test shape (merged POS): {}\n".format(application_test_df.shape))

    # merge credit data
    credit_df = pd.read_csv("../../all/credit_card_balance.csv")
    edit_credit_df = feature_credit_balance(credit_df)
    application_train_df = application_train_df.merge(edit_credit_df, on=["SK_ID_CURR"], how="left")
    application_test_df = application_test_df.merge(edit_credit_df, on=["SK_ID_CURR"], how="left")
    print("train shape (merged credit): {}".format(application_train_df.shape))
    print("test shape (merged credit): {}\n".format(application_test_df.shape))

    # merge installments data
    inst_df = pd.read_csv("../../all/installments_payments.csv")
    edit_inst_df = feature_installments(inst_df)
    application_train_df = application_train_df.merge(edit_inst_df, on=["SK_ID_CURR"], how="left")
    application_test_df = application_test_df.merge(edit_inst_df, on=["SK_ID_CURR"], how="left")
    print("train shape (merged installments): {}".format(application_train_df.shape))
    print("test shape (merged installments): {}\n".format(application_test_df.shape))

    application_train_df.to_csv("../../all/train.csv", index=False)
    application_test_df.to_csv("../../all/test.csv", index=False)
    print("DONE! save merge data!\n\n")

    return application_train_df, application_test_df


def make_dataset(train_df, test_df):
    print("\n****************************************************")
    print("make dataset")
    print("Train data features shape: {}".format(train_df.shape))
    print("Test data features shape: {}".format(test_df.shape))
    
    # One Hot Encoding
    # yes/No -> 1/0
    le = LabelEncoder()
    le_count = 0

    # only label encode those variables with 2 or less categories
    for col in train_df:
        if train_df[col].dtype == "object":
            # if 2 or fewer unique categories
            if len(list(train_df[col].unique())) <= 2:
                print(col)
                # Train on the training data
                le.fit(train_df[col])
                # Transeform noth training and testing data
                train_df[col] = le.transform(train_df[col])
                test_df[col] = le.transform(test_df[col])

                # keep track of how many columns were label encoded
                le_count += 1

    print("{} columns were labeld encoded.\n".format(le_count))
    
    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)

    print("Train data features shape: {}".format(train_df.shape))
    print("Test data features shape: {}".format(test_df.shape))
    
    target = train_df["TARGET"]
    train_df, test_df = train_df.align(test_df, join="inner", axis=1)
    print("Train data features shape: {}".format(train_df.shape))
    print("Test data features shape: {}".format(test_df.shape))
    
    if "TARGET" in train_df:
        train_df = train_df.drop("TARGET", axis=1)

    features = list(train_df.columns)

    # Median imputation of missing values
    imputer = Imputer(strategy="median")
    print("DONE: Imputation")
    
    # Scale each feature 0 - 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    print("DONE: Scale")

    # Fit on the training data
    imputer.fit(train_df)
    print("DONE: Fit")

    # Transform both training and test data
    train_df = imputer.transform(train_df.astype(np.float32))
    test_df = imputer.transform(test_df.astype(np.float32))
    print("DONE: Transform\n")
    
    # Repeat with the scaler
    scaler.fit(train_df.astype(np.int))
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)
    print("DONE: Scalar Transform")

    print("Train data features shape: {}".format(train_df.shape))
    print("Test data features shape: {}".format(test_df.shape))
    
    np.save("../../all/train_X", train_df)
    np.save("../../all/train_target", target)
    np.save("../../all/test", test_df)
    print("DONE! save train.npy, target.npy, test.npy") 
    
    # X_train, X_val, y_train, y_val = train_test_split(train_df, target, test_size=0.2, random_state=0)

    # print("X_train shape: {}".format(X_train.shape))
    # print("X_val shape: {}".format(X_val.shape))
    # print("y_train shape: {}".format(y_train.shape))
    # print("y_val shape: {}".format(y_val.shape))
    
    # np.save("../../all/X_train", X_train)
    # np.save("../../all/X_val", X_val)
    # np.save("../../all/y_train", y_train)
    # np.save("../../all/y_val", y_val)
    
    print("\n*********** DONE! ***************")


if __name__ == "__main__":
    application_train_path = "../../all/application_train.csv" 
    application_test_path =  "../../all/application_test.csv"
    train_df, test_df = merge_application(application_train_path, application_test_path)

    make_dataset(train_df, test_df)

