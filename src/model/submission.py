#! /usr/bin/env python

import numpy as np
import pandas as pd
#from model_mlp import mlp_main
#from model_lightgbm import lightgbm

def submission(mlp_pred, gbm_pred):
    merge_pred = 0.2*mlp_pred + 0.8*gbm_pred
    # output format
    submit_dictlst = {"SK_ID_CURR":[], "TARGET":[]}
    id_lst = pd.read_csv("../../all/application_test.csv")["SK_ID_CURR"]
    for i in range(len(id_lst)):
        submit_dictlst["SK_ID_CURR"].append(id_lst[i])
        submit_dictlst["TARGET"].append(merge_pred[i])

    pd.DataFrame(submit_dictlst).to_csv("../../all/submission.csv", index=False)
    print("DONE!")

if __name__ == "__main__":
    train_X_path = "../../all/train_X.npy"
    train_y_path = "../../all/train_target.npy"
    test_path = "../../all/test.npy"
    reuse_model_path = "../../all/tensor_model/"
    # mlp_pred =  mlp_main(train_X_path, train_y_path, test_path, reuse_model_path, learning=False)
    # gbm_pred = lightgbm(train_X_path, train_y_path, test_path)

    # MLP prediction
    mlp_pred = np.load("../../all/out_mlp.npy")
    # lightGBM prediction
    gbm_pred = np.load("../../all/out_gbm.npy")

    submission(mlp_pred, gbm_pred)
