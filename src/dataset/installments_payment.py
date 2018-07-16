import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc

def feature_installments(inst_df):
    print("\n****************************************************") 
    print("Feature Engineering: installment data\n")

    # 各ID_PREVについて、最新のものを取得
    grp = inst_df[["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]].groupby(by=["SK_ID_PREV"])["NUM_INSTALMENT_NUMBER"].max().reset_index().rename(index=str, columns={"NUM_INSTALMENT_NUMBER":"LATEST_INSTALMENT_NUMBER"})
    inst_df = inst_df.merge(grp, on="SK_ID_PREV", how="left")
    inst_df = inst_df[inst_df["NUM_INSTALMENT_NUMBER"]==inst_df["LATEST_INSTALMENT_NUMBER"]]
    inst_df = inst_df.drop("NUM_INSTALMENT_NUMBER", axis=1).reset_index(drop=True)
    inst_df = inst_df[~inst_df["SK_ID_PREV"].duplicated()]
    print("installments data shape: {}".format(inst_df.shape))
    
    # AMT_INSTALMENTとAMT_PAYMENTの差分
    inst_df["DIFF_AMT_INSTALLMENT_PAYMENT"] = inst_df["AMT_INSTALMENT"] - inst_df["AMT_PAYMENT"]
    print("installments data shape: {}".format(inst_df.shape))

    # DAYS_INSTALMENTとDAYS_ENTRY_PAYMENTの差分
    inst_df["DIFF_DAYS_INSTALLMENT_PAYMENT"] = inst_df["DAYS_INSTALMENT"] - inst_df["DAYS_ENTRY_PAYMENT"]
    print("installments data shape: {}".format(inst_df.shape))
    
    # ID_CURR毎のID_PREVの数をカウント
    grp = inst_df[["SK_ID_PREV", "SK_ID_CURR"]].groupby(by=["SK_ID_CURR"])["SK_ID_PREV"].count().reset_index().rename(index=str, columns={"SK_ID_PREV":"PREV_COUNT"})
    inst_df = inst_df.merge(grp, on="SK_ID_CURR", how="left")
    print("installments data shape: {}".format(inst_df.shape))
    
    # ID_CURRに対して、複数のID_PREVが存在する
    # 各統計値のID_CURRについての平均を計算し、uniqueにする
    grp = inst_df[["SK_ID_CURR", "NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT", \
                   "LATEST_INSTALMENT_NUMBER", "DIFF_AMT_INSTALLMENT_PAYMENT", "DIFF_DAYS_INSTALLMENT_PAYMENT"]].groupby(by=["SK_ID_CURR"]).mean().reset_index().rename(index=str, columns={\
                    "NUM_INSTALMENT_VERSION":"NUM_INSTALMENT_VERSION_MEAN", "DAYS_INSTALMENT":"DAYS_INSTALMENT_MEAN",
                    "DAYS_ENTRY_PAYMENT":"DAYS_ENTRY_PAYMENT_MEAN",  "AMT_INSTALMENT": "AMT_INSTALMENT_MEAN",
                    "AMT_PAYMENT":"AMT_PAYMENT_MEAN","LATEST_INSTALMENT_NUMBER":"LATEST_INSTALMENT_NUMBER_MEAN",
                    "DIFF_AMT_INSTALLMENT_PAYMENT":"DIFF_AMT_INSTALLMENT_PAYMENT_MEAN", \
                    "DIFF_DAYS_INSTALLMENT_PAYMENT":"DIFF_DAYS_INSTALLMENT_PAYMENT_MEAN"})
    inst_df = inst_df.merge(grp, on="SK_ID_CURR", how="left")
    drop_lst = ["SK_ID_PREV","NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT", \
                   "LATEST_INSTALMENT_NUMBER", "DIFF_AMT_INSTALLMENT_PAYMENT", "DIFF_DAYS_INSTALLMENT_PAYMENT"]
    inst_df = inst_df.drop(drop_lst, axis=1)
    inst_df = inst_df[~inst_df["SK_ID_CURR"].duplicated()]
    inst_df = inst_df.fillna(0)
    print("installments data shape: {}\n".format(inst_df.shape))
    
    # check duplication
    if len(inst_df[inst_df["SK_ID_CURR"].duplicated()]) == 0:
        print("No duplication!\n")
    else:
        print("ERROR: duplicated!\n")
    
    # check NaN
    print("check NaN")
    print("column name / number of NaN")
    print(np.sum(inst_df.isnull()))
    
    print("\nDONE!  feature engineering of installments payment")
    print("****************************************************\n\n")
    
    return inst_df