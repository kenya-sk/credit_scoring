import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc


def feature_pos(pos_df):
    print("\n****************************************************") 
    print("Feature Engineering: POS data\n")
    
    
    # current ID毎のprev IDの数をカウント
    grp = pos_df[["SK_ID_PREV", "SK_ID_CURR"]].groupby(by=["SK_ID_CURR"])["SK_ID_PREV"].count().reset_index().rename(index=str, columns={"SK_ID_PREV":"PREV_COUNT"})
    pos_df = pos_df.merge(grp, on="SK_ID_CURR", how="left")
    print("POS data shape: {}".format(pos_df.shape))

    # SK_DPD, SK_DPD_DEFのsum, max, mean
    dpd_max_grp = pos_df[["SK_ID_PREV", "SK_DPD"]].groupby(by=["SK_ID_PREV"])["SK_DPD"].max().reset_index().rename(index=str, columns={"SK_DPD":"SK_DPD_MAX"})
    pos_df = pos_df.merge(dpd_max_grp, on="SK_ID_PREV", how="left")
    dpd_sum_grp = pos_df[["SK_ID_PREV", "SK_DPD"]].groupby(by=["SK_ID_PREV"])["SK_DPD"].sum().reset_index().rename(index=str, columns={"SK_DPD":"SK_DPD_SUM"})
    pos_df = pos_df.merge(dpd_sum_grp, on="SK_ID_PREV", how="left")
    dpd_mean_grp = pos_df[["SK_ID_PREV", "SK_DPD"]].groupby(by=["SK_ID_PREV"])["SK_DPD"].mean().reset_index().rename(index=str, columns={"SK_DPD":"SK_DPD_MEAN"})
    pos_df = pos_df.merge(dpd_mean_grp, on="SK_ID_PREV", how="left")

    dpd_def_max_grp = pos_df[["SK_ID_PREV", "SK_DPD_DEF"]].groupby(by=["SK_ID_PREV"])["SK_DPD_DEF"].max().reset_index().rename(index=str, columns={"SK_DPD_DEF":"SK_DPD_DEF_MAX"})
    pos_df = pos_df.merge(dpd_def_max_grp, on="SK_ID_PREV", how="left")
    dpd_def_sum_grp = pos_df[["SK_ID_PREV", "SK_DPD_DEF"]].groupby(by=["SK_ID_PREV"])["SK_DPD_DEF"].sum().reset_index().rename(index=str, columns={"SK_DPD_DEF":"SK_DPD_DEF_SUM"})
    pos_df = pos_df.merge(dpd_def_sum_grp, on="SK_ID_PREV", how="left")
    dpd_def_mean_grp = pos_df[["SK_ID_PREV", "SK_DPD_DEF"]].groupby(by=["SK_ID_PREV"])["SK_DPD_DEF"].mean().reset_index().rename(index=str, columns={"SK_DPD_DEF":"SK_DPD_DEF_MEAN"})
    pos_df = pos_df.merge(dpd_def_mean_grp, on="SK_ID_PREV", how="left")
    print("POS data shape: {}".format(pos_df.shape))

    # それぞれ最新のものを用いることで、SK_ID_PREVをuniqueにする
    latest_grp= pos_df[["SK_ID_PREV", "MONTHS_BALANCE"]].groupby(by=["SK_ID_PREV"]).max().reset_index().rename(index=str, columns={"MONTHS_BALANCE":"LATEST_BALANCE"})
    pos_df = pos_df.merge(latest_grp, on="SK_ID_PREV", how="left")
    pos_df = pos_df[pos_df["MONTHS_BALANCE"]==pos_df["LATEST_BALANCE"]]
    pos_df = pos_df.drop("MONTHS_BALANCE", axis=1)
    print("POS data shape: {}".format(pos_df.shape))
    
    drop_columns_lst = ["SK_DPD", "SK_DPD_DEF"]
    pos_df = pos_df.drop(drop_columns_lst, axis=1)
    print("POS data shape: {}".format(pos_df.shape))
    
    # STATUSをダミー変数に変換
    pos_df = pd.get_dummies(pos_df)
    
    # ID_CURRに対して、複数のID_PREVが存在する
    # 各統計値のID_CURRについての平均を計算し、uniqueにする

    grp = pos_df[["SK_ID_CURR", "CNT_INSTALMENT","CNT_INSTALMENT_FUTURE", "SK_DPD_MAX", "SK_DPD_SUM", "SK_DPD_MEAN", "SK_DPD_DEF_MAX", "SK_DPD_DEF_SUM", "SK_DPD_DEF_MEAN", "LATEST_BALANCE",\
                  "NAME_CONTRACT_STATUS_Active", "NAME_CONTRACT_STATUS_Amortized debt", "NAME_CONTRACT_STATUS_Approved", "NAME_CONTRACT_STATUS_Canceled",\
                  "NAME_CONTRACT_STATUS_Completed", "NAME_CONTRACT_STATUS_Demand", "NAME_CONTRACT_STATUS_Returned to the store", \
                  "NAME_CONTRACT_STATUS_Signed"]].groupby(by=["SK_ID_CURR"]).mean().reset_index().rename(index=str, columns={\
                           "CNT_INSTALMENT":"CNT_INSTALMENT_MEAN", "CNT_INSTALMENT_FUTURE":"CNT_INSTALMENT_FUTURE_MEAN",
                           "SK_DPD_MAX":"SK_DPD_MAX_MEAN", "SK_DPD_SUM":"SK_DPD_SUM_MEAN",
                           "SK_DPD_MEAN":"SK_DPD_MEAN_MEAN", "SK_DPD_DEF_MAX":"SK_DPD_DEF_MAX_MEAN",
                           "SK_DPD_DEF_SUM":"SK_DPD_DEF_SUM_MEAN", "SK_DPD_DEF_MEAN":"SK_DPD_DEF_MEAN_MEAN",
                           "LATEST_BALANCE":"LATEST_BALANCE_MEAN", "NAME_CONTRACT_STATUS_Active":"NAME_CONTRACT_STATUS_Active_MEAN",
                           "NAME_CONTRACT_STATUS_Amortized debt":"NAME_CONTRACT_STATUS_Amortized debt_MEAN",
                           "NAME_CONTRACT_STATUS_Approved":"NAME_CONTRACT_STATUS_Approved_MEAN",
                           "NAME_CONTRACT_STATUS_Canceled":"NAME_CONTRACT_STATUS_Canceled_MEAN",
                           "NAME_CONTRACT_STATUS_Completed":"NAME_CONTRACT_STATUS_Completed_MEAN",
                           "NAME_CONTRACT_STATUS_Demand":"NAME_CONTRACT_STATUS_Demand_MEAN",
                           "NAME_CONTRACT_STATUS_Returned to the store":"NAME_CONTRACT_STATUS_Returned to the store_MEAN",
                           "NAME_CONTRACT_STATUS_Signed":"NAME_CONTRACT_STATUS_Signed_MEAN"})

    pos_df = pos_df.merge(grp, on="SK_ID_CURR", how="left")
    drop_lst = ["SK_ID_PREV","CNT_INSTALMENT","CNT_INSTALMENT_FUTURE", "SK_DPD_MAX", "SK_DPD_SUM", "SK_DPD_MEAN", "SK_DPD_DEF_MAX", "SK_DPD_DEF_SUM", "SK_DPD_DEF_MEAN", "LATEST_BALANCE",
                    "NAME_CONTRACT_STATUS_Active", "NAME_CONTRACT_STATUS_Amortized debt", "NAME_CONTRACT_STATUS_Approved", "NAME_CONTRACT_STATUS_Canceled",
                    "NAME_CONTRACT_STATUS_Completed", "NAME_CONTRACT_STATUS_Demand", "NAME_CONTRACT_STATUS_Returned to the store",
                    "NAME_CONTRACT_STATUS_Signed"]
    pos_df = pos_df.drop(drop_lst, axis=1)
    pos_df = pos_df[~pos_df["SK_ID_CURR"].duplicated()].reset_index(drop=True)
    print("POS data shape: {}\n".format(pos_df.shape))
    
    # CHECK: fill 0 or median
    pos_df[["CNT_INSTALMENT_MEAN" ,"CNT_INSTALMENT_FUTURE_MEAN"]] = pos_df[["CNT_INSTALMENT_MEAN" ,"CNT_INSTALMENT_FUTURE_MEAN"]].fillna(0)
    
    # check duplication
    if len(pos_df[pos_df["SK_ID_CURR"].duplicated()]) == 0:
        print("No duplication!\n")
    else:
        print("ERROR: duplicated!\n")
        
    print("check NaN")
    print("column name / number of NaN")
    print(np.sum(pos_df.isnull()))
        
    print("DONE! feature engineering of POS")
    print("****************************************************\n\n")

    return pos_df