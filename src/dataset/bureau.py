import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc

def feature_bureau(edit_bureau_df):
    def f1(x):
        y=0 if x=="Closed" else 1
        return y

    def binalization(x):
        y=0 if x<0 else 1
        return y

    # ID_CURRごとの過去のクレジット利用回数
    user_grp = edit_bureau_df[["SK_ID_CURR", "DAYS_CREDIT"]].groupby(by=["SK_ID_CURR"])["DAYS_CREDIT"].count().reset_index().rename(index=str, columns={"DAYS_CREDIT":"BUREAU_LOAN_COUNT"})
    edit_bureau_df = edit_bureau_df.merge(user_grp, on=["SK_ID_CURR"], how="left")

    print("bureau data shape: {}".format(edit_bureau_df.shape))

    # ID_CURRごとの利用ローンの種類数
    user_grp = edit_bureau_df[["SK_ID_CURR", "CREDIT_TYPE"]].groupby(by=["SK_ID_CURR"])["CREDIT_TYPE"].nunique().reset_index().rename(index=str, columns={"CREDIT_TYPE":"BUREAU_LOAN_TYPES"})
    edit_bureau_df = edit_bureau_df.merge(user_grp, on=["SK_ID_CURR"], how="left")

    print("bureau data shape: {}".format(edit_bureau_df.shape))

    # ID_CURRごとの各ローンあたりの平均利用回数
    edit_bureau_df["AVERAGE_LOAN_TYPE"] = edit_bureau_df["BUREAU_LOAN_COUNT"]/edit_bureau_df["BUREAU_LOAN_TYPES"]
    edit_bureau_df = edit_bureau_df.drop(["BUREAU_LOAN_COUNT", "BUREAU_LOAN_TYPES"], axis=1)
    gc.collect()
    
    print("burau data shape: {}".format(edit_bureau_df.shape))

    # BUREAU DATAを元にした、ID_CURRごとのActiveなローンの数
    edit_bureau_df["CREDIT_ACTIVE_BINARY"] = edit_bureau_df["CREDIT_ACTIVE"]
    edit_bureau_df["CREDIT_ACTIVE_BINARY"] = edit_bureau_df.apply(lambda x : f1(x.CREDIT_ACTIVE), axis=1)
    # calculate mean number of loans that are ACTIVE per CUSTOMER
    active_grp = edit_bureau_df.groupby(by=["SK_ID_CURR"])["CREDIT_ACTIVE_BINARY"].mean().reset_index().rename(index=str, columns={"CREDIT_ACTIVE_BINARY":"ACTIVE_LOANS_PERCENTAGE"})
    edit_bureau_df = edit_bureau_df.merge(active_grp, on=["SK_ID_CURR"], how="left")
    edit_bureau_df = edit_bureau_df.drop("CREDIT_ACTIVE_BINARY", axis=1)
    gc.collect()
    edit_bureau_df["ACTIVE_LOANS_PERCENTAGE"] = 100*edit_bureau_df["ACTIVE_LOANS_PERCENTAGE"]

    print("burau data shape: {}".format(edit_bureau_df.shape))

    # AVERAGE NUMBER OF DAYS BETWEEN SUCCESSIVE PAST APPLICATIONS FOR EACH CUSTOMER
    #groupby each customer and sort values of DAYS_CREDIT in ascending order
    grp = edit_bureau_df[["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT"]].groupby(by=["SK_ID_CURR"])
    grouped_df = grp.apply(lambda x: x.sort_values(["DAYS_CREDIT"], ascending=False)).reset_index(drop=True)
    print("Grouping and Sorting done")
    # calculate difference between the number of Days
    grouped_df["DAYS_CREDIT1"] = grouped_df["DAYS_CREDIT"]*-1
    grouped_df["DAYS_DIFF"] = grouped_df.groupby(by=["SK_ID_CURR"])["DAYS_CREDIT1"].diff()
    grouped_df["DAYS_DIFF"] = grouped_df["DAYS_DIFF"].fillna(0).astype("uint32")
    grouped_df =  grouped_df.groupby(by=["SK_ID_CURR"])["DAYS_DIFF"].mean().reset_index().rename(index=str, columns={"DAYS_DIFF":"DAYS_DIFF_MEAN"})
    print("Diffenrence days calculated")
    edit_bureau_df = edit_bureau_df.merge(grouped_df, on=["SK_ID_CURR"], how="left")
    edit_bureau_df["DAYS_DIFF_MEAN"] = edit_bureau_df["DAYS_DIFF_MEAN"].fillna(0.0)
    print("diffenrence in Datas between Previous CB application in CALCULATED")

    print("bureau data shape: {}".format(edit_bureau_df.shape))

    # % OF LOANS PER CUSTOMER WHERE END DATE FOR CREDIT ID PAST
    edit_bureau_df["CREDIT_ENDDATE_BINARY"] = edit_bureau_df["DAYS_CREDIT_ENDDATE"]
    edit_bureau_df["CREDIT_ENDDATE_BINARY"] = edit_bureau_df.apply(lambda x: binalization(x.DAYS_CREDIT_ENDDATE), axis=1)
    print("New Binary Column calculated")
    dedline_grp = edit_bureau_df.groupby(by=["SK_ID_CURR"])["CREDIT_ENDDATE_BINARY"].mean().reset_index().rename(index=str, columns={"CREDIT_ENDDATE_BINARY":"CREDIT_ENDDATE_PERCENTAGE"})
    edit_bureau_df = edit_bureau_df.merge(dedline_grp, on=["SK_ID_CURR"], how="left")
    edit_bureau_df = edit_bureau_df.drop("CREDIT_ENDDATE_BINARY", axis=1)
    gc.collect()
    edit_bureau_df["CREDIT_ENDDATE_PERCENTAGE"] = np.abs(100*edit_bureau_df["CREDIT_ENDDATE_PERCENTAGE"])

    print("burau data shape: {}".format(edit_bureau_df.shape))

    # ID_CURRごとのクレジット超過の割合
    edit_bureau_df["AMT_CREDIT_SUM_DEBT"] = edit_bureau_df["AMT_CREDIT_SUM_DEBT"].fillna(0.0)
    edit_bureau_df["AMT_CREDIT_SUM"] = edit_bureau_df["AMT_CREDIT_SUM"].fillna(0.0)
    debt_grp1 = edit_bureau_df[["SK_ID_CURR", "AMT_CREDIT_SUM_DEBT"]].groupby(by=["SK_ID_CURR"])["AMT_CREDIT_SUM_DEBT"].sum().reset_index().rename(index=str, columns={"AMT_CREDIT_SUM_DEBT":"TOTAL_CUSTOMER_DEBT"})
    debt_grp2 = edit_bureau_df[["SK_ID_CURR", "AMT_CREDIT_SUM"]].groupby(by=["SK_ID_CURR"])["AMT_CREDIT_SUM"].sum().reset_index().rename(index=str, columns={"AMT_CREDIT_SUM":"TOTAL_CUSTOMER_CREDIT"})
    edit_bureau_df = edit_bureau_df.merge(debt_grp1, on=["SK_ID_CURR"], how="left")
    edit_bureau_df = edit_bureau_df.merge(debt_grp2, on=["SK_ID_CURR"], how="left")
    del debt_grp1, debt_grp2
    gc.collect()
    edit_bureau_df["DEBT_CREDIT_RATIO"] = edit_bureau_df["TOTAL_CUSTOMER_DEBT"]/edit_bureau_df["TOTAL_CUSTOMER_CREDIT"]
    edit_bureau_df["DEBT_CREDIT_RATIO"] = edit_bureau_df["DEBT_CREDIT_RATIO"].fillna(0.0)
    edit_bureau_df["DEBT_CREDIT_RATIO"] = np.abs(100*edit_bureau_df["DEBT_CREDIT_RATIO"])
    edit_bureau_df["DEBT_CREDIT_RATIO"] = edit_bureau_df["DEBT_CREDIT_RATIO"].replace(np.inf, 0.0)
    del edit_bureau_df["TOTAL_CUSTOMER_DEBT"], edit_bureau_df["TOTAL_CUSTOMER_CREDIT"]
    gc.collect()

    print("burau data shpae: {}".format(edit_bureau_df.shape))

    # ID_CURRごとの借金超過の割合
    edit_bureau_df["AMT_CREDIT_SUM_DEBT"] = edit_bureau_df["AMT_CREDIT_SUM_DEBT"].fillna(0)
    edit_bureau_df["AMT_CREDIT_SUM_OVERDUE"] = edit_bureau_df["AMT_CREDIT_SUM_OVERDUE"].fillna(0)
    overdue_grp1 = edit_bureau_df[["SK_ID_CURR", "AMT_CREDIT_SUM_DEBT"]].groupby(by=["SK_ID_CURR"])["AMT_CREDIT_SUM_DEBT"].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    overdue_grp2 = edit_bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    edit_bureau_df = edit_bureau_df.merge(overdue_grp1, on=["SK_ID_CURR"], how="left")
    edit_bureau_df = edit_bureau_df.merge(overdue_grp2, on=["SK_ID_CURR"], how="left")
    del overdue_grp1, overdue_grp2
    gc.collect()
    edit_bureau_df["OVERDUE_DEBT_RATIO"] = edit_bureau_df["TOTAL_CUSTOMER_OVERDUE"]/edit_bureau_df["TOTAL_CUSTOMER_DEBT"]
    edit_bureau_df["OVERDUE_DEBT_RATIO"] = edit_bureau_df["OVERDUE_DEBT_RATIO"].replace(np.inf, 0.0)
    edit_bureau_df["OVERDUE_DEBT_RATIO"] = edit_bureau_df["OVERDUE_DEBT_RATIO"].fillna(0.0)
    edit_bureau_df["OVERDUE_DEBT_RATIO"] = np.abs(100*edit_bureau_df["OVERDUE_DEBT_RATIO"])
    del edit_bureau_df["TOTAL_CUSTOMER_OVERDUE"], edit_bureau_df["TOTAL_CUSTOMER_DEBT"]
    gc.collect()

    print("burau data shape: {}".format(edit_bureau_df.shape))

    # ID_CURRごとのローン延長の割合
    edit_bureau_df["CNT_CREDIT_PROLONG"] = edit_bureau_df["CNT_CREDIT_PROLONG"].fillna(0)
    prolong_grp = edit_bureau_df[["SK_ID_CURR", "CNT_CREDIT_PROLONG"]].groupby(by=["SK_ID_CURR"])["CNT_CREDIT_PROLONG"].mean().reset_index().rename(index=str, columns={"CNT_CREDIT_PROLONG":"AVG_CREDITDAYS_PROLONGED"})
    edit_bureau_df = edit_bureau_df.merge(prolong_grp, on=["SK_ID_CURR"], how="left")

    print("burau data shape: {}".format(edit_bureau_df.shape))

    # 平均値などのカラムを残すことで,SK_ID_CURRをユニークにする
    drop_columns = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY',
       'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
       'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
       'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
       'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE',
       'AMT_ANNUITY']

    edit_bureau_df = edit_bureau_df.drop(drop_columns, axis=1)
    print("droped no need columns")
    print("burau data shape: {}\n".format(edit_bureau_df.shape))

    edit_bureau_df = edit_bureau_df[~edit_bureau_df["SK_ID_CURR"].duplicated()].reset_index(drop=True)
    # check duplication
    if len(edit_bureau_df[edit_bureau_df["SK_ID_CURR"].duplicated()]) == 0:
        print("No duplication!\n")
    else:
        print("ERROR: duplicated!\n")

    print("check NaN")
    print("column name / number of NaN")
    print(np.sum(edit_bureau_df.isnull()))

    return edit_bureau_df



def bureau_latest(bureau_balance_df):
    # 各SK_ID_BUREAUの最新のデータをのみを取ってくる
    grp = bureau_balance_df[["SK_ID_BUREAU", "MONTHS_BALANCE"]].groupby(by=["SK_ID_BUREAU"])["MONTHS_BALANCE"].max().reset_index().rename(index=str, columns={"MONTHS_BALANCE":"MONTHS_BALANCE_LATEST"})
    bureau_balance_df = bureau_balance_df.merge(grp, on=["SK_ID_BUREAU"], how="left")
    bureau_balance_df["STATUS_LATEST"] = bureau_balance_df[bureau_balance_df["MONTHS_BALANCE"] == bureau_balance_df["MONTHS_BALANCE_LATEST"]]["STATUS"]

    bureau_balance_df = bureau_balance_df.drop(["MONTHS_BALANCE", "STATUS"], axis=1)
    bureau_balance_df = bureau_balance_df[~bureau_balance_df["SK_ID_BUREAU"].duplicated()]
    print("bureau balance data shape: {}\n".format(bureau_balance_df.shape))

    print("check NaN")
    print("column name / number of NaN")
    print(np.sum(bureau_balance_df.isnull()))

    return bureau_balance_df


def merge_bureau(bureau_df, bureau_balance_df):
    print("\n****************************************************") 
    print("\n\nFeature Engineering: bureau data")

    edit_bureau_df = feature_bureau(bureau_df)
    latest_bureau_df = bureau_latest(bureau_balance_df)
    merge_bureau_df = edit_bureau_df.merge(latest_bureau_df, on="SK_ID_BUREAU", how="left")
    merge_bureau_df["MONTHS_BALANCE_LATEST"] = merge_bureau_df["MONTHS_BALANCE_LATEST"].fillna(0)
    merge_bureau_df["STATUS_LATEST"] = merge_bureau_df["STATUS_LATEST"].fillna("X")
    merge_bureau_df = merge_bureau_df.drop("SK_ID_BUREAU", axis=1)
    print("merge bureau data shape: {}".format(merge_bureau_df.shape))

    # check duplication
    if len(merge_bureau_df[merge_bureau_df["SK_ID_CURR"].duplicated()]) == 0:
        print("No duplication!\n")
    else:
        print("ERROR: duplicated!\n")

    print("check NaN")
    print("column name / number of NaN")
    print(np.sum(merge_bureau_df.isnull()))
    
    print("\nDone! feature engineering of bureau\n\n")
    print("****************************************************\n\n")
    
    return merge_bureau_df