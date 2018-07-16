import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gc

def feature_credit_balance(credit_df):
    def f1(x1, x2):
        balance = x1.max()
        limit = x2.max()
        return (balance/limit)
    
    def f2(DPD):
        x = DPD.tolist()
        c = 0
        for i,j in enumerate(x):
            if j != 0:
                c += 1
        return c 
    
    def f3(min_pay, total_pay):
        M = min_pay.tolist()
        T = total_pay.tolist()
        P = len(M)
        c = 0 
        for i in range(len(M)):
            if T[i] < M[i]:
                c += 1  
        return (100*c)/P
                           
    print("\n****************************************************") 
    print("Feature Engineering: credit data\n")
    
    # ID_CURRあたりのID_PREVの総数
    grp = credit_df.groupby(by = ['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index = str, columns = {'SK_ID_PREV': 'NO_LOANS'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp 
    gc.collect()
    print("credit data shape: {}".format(credit_df.shape))
    
    # 借入金の支払額
    grp = credit_df.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index = str, columns = {'CNT_INSTALMENT_MATURE_CUM': 'NO_INSTALMENTS'})
    grp1 = grp.groupby(by = ['SK_ID_CURR'])['NO_INSTALMENTS'].sum().reset_index().rename(index = str, columns = {'NO_INSTALMENTS': 'TOTAL_INSTALMENTS'})
    credit_df = credit_df.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
    del grp, grp1
    gc.collect()
    
    credit_df['INSTALLMENTS_PER_LOAN'] = (credit_df['TOTAL_INSTALMENTS']/credit_df['NO_LOANS']).astype('uint32')
    credit_df['INSTALLMENTS_PER_LOAN'] = credit_df['INSTALLMENTS_PER_LOAN'].replace(np.inf, 0.0)
    del credit_df['TOTAL_INSTALMENTS']
    del credit_df['NO_LOANS']
    gc.collect()
    print("credit data shape: {}".format(credit_df.shape))

    # ID_CURRあたりのクレジット限度額
    credit_df['AMT_CREDIT_LIMIT_ACTUAL1'] = credit_df['AMT_CREDIT_LIMIT_ACTUAL']
    grp = credit_df.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(lambda x: f1(x.AMT_BALANCE, x.AMT_CREDIT_LIMIT_ACTUAL1)).reset_index().rename(index = str, columns = {0: 'CREDIT_LOAD1'})
    del credit_df['AMT_CREDIT_LIMIT_ACTUAL1']
    gc.collect()

    grp1 = grp.groupby(by = ['SK_ID_CURR'])['CREDIT_LOAD1'].mean().reset_index().rename(index = str, columns = {'CREDIT_LOAD1': 'CREDIT_LOAD'})

    credit_df = credit_df.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
    credit_df["CREDIT_LOAD"] = credit_df["CREDIT_LOAD"].replace(np.inf, 0.0)
    del grp, grp1
    gc.collect()
    print("credit data shape: {}".format(credit_df.shape))
    
    # ID_CURRあたりのDPD
    grp = credit_df.groupby(by = ['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: f2(x.SK_DPD)).reset_index().rename(index = str, columns = {0: 'NO_DPD'})
    grp1 = grp.groupby(by = ['SK_ID_CURR'])['NO_DPD'].mean().reset_index().rename(index = str, columns = {'NO_DPD' : 'DPD_COUNT'})
    credit_df = credit_df.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
    del grp1
    del grp 
    gc.collect()
    print("credit data shape: {}".format(credit_df.shape))
    
    # 過去の一日あたりのクレジット利用頻度
    grp = credit_df.groupby(by= ['SK_ID_CURR'])['SK_DPD'].mean().reset_index().rename(index = str, columns = {'SK_DPD': 'AVG_DPD'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp 
    gc.collect()
    print("credit data shape: {}".format(credit_df.shape))
    
    # 支払最低額が支払われていない割合
    grp = credit_df.groupby(by = ['SK_ID_CURR']).apply(lambda x: f3(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index = str, columns = { 0 : 'PERCENTAGE_MISSED_PAYMENTS'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    credit_df["PERCENTAGE_MISSED_PAYMENTS"] = credit_df["PERCENTAGE_MISSED_PAYMENTS"].replace(np.inf, 0.0)
    del grp 
    gc.collect()
    print("credit data shape: {}".format(credit_df.shape))
    
    # 支払と引き出しの比率
    grp = credit_df.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_ATM_CURRENT' : 'DRAWINGS_ATM'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp
    gc.collect()
    
    grp = credit_df.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'DRAWINGS_TOTAL'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp
    gc.collect()
    
    credit_df['CASH_CARD_RATIO1'] = (credit_df['DRAWINGS_ATM']/credit_df['DRAWINGS_TOTAL'])*100
    credit_df['CASH_CARD_RATIO1'] = credit_df['CASH_CARD_RATIO1'].replace(np.inf, 0.0)
    del credit_df['DRAWINGS_ATM']
    del credit_df['DRAWINGS_TOTAL']
    gc.collect()

    grp = credit_df.groupby(by = ['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'CASH_CARD_RATIO1' : 'CASH_CARD_RATIO'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp 
    gc.collect()

    del credit_df['CASH_CARD_RATIO1']
    gc.collect()
    print("credit data shape: {}".format(credit_df.shape))
    
    # ID_CURRあたりの引き出しの割合
    grp = credit_df.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'TOTAL_DRAWINGS'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp
    gc.collect()

    grp = credit_df.groupby(by = ['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'CNT_DRAWINGS_CURRENT' : 'NO_DRAWINGS'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp
    gc.collect()

    credit_df['DRAWINGS_RATIO1'] = (credit_df['TOTAL_DRAWINGS']/credit_df['NO_DRAWINGS'])*100
    del credit_df['TOTAL_DRAWINGS']
    del credit_df['NO_DRAWINGS']
    gc.collect()
    credit_df['DRAWINGS_RATIO1'] = credit_df['DRAWINGS_RATIO1'].replace(np.inf, 0.0)

    grp = credit_df.groupby(by = ['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'DRAWINGS_RATIO1' : 'DRAWINGS_RATIO'})
    credit_df = credit_df.merge(grp, on = ['SK_ID_CURR'], how = 'left')
    del grp 
    gc.collect()

    del credit_df['DRAWINGS_RATIO1']
    print("credit data shape: {}".format(credit_df.shape))
    
    # 不要なカラムをdrop
    drop_lst = ["SK_ID_PREV","MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT", \
                     "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL",\
                     "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE", "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT",\
                     "CNT_DRAWINGS_POS_CURRENT", "CNT_INSTALMENT_MATURE_CUM", "NAME_CONTRACT_STATUS", "SK_DPD",\
                     "SK_DPD_DEF"]
    credit_df = credit_df.drop(drop_lst, axis=1)
    print("credit data shape: {}\n".format(credit_df.shape))
        
    
    # ID_CURRをuniqueにするため、重複を省く
    credit_df = credit_df[~credit_df["SK_ID_CURR"].duplicated()].reset_index(drop=True)

    # check duplication
    if len(credit_df[credit_df["SK_ID_CURR"].duplicated()]) == 0:
        print("No duplication!\n")
    else:
        print("ERROR: duplicated!\n")
        
    # fill NaN
    credit_df = credit_df.fillna(0)
    print("check NaN.")
    print("column name / number of NaN")
    print(np.sum(credit_df.isnull()))
    
    # -inf -> 0
    credit_df = credit_df.replace(-np.inf, 0.0)
    
    print("\nDONE: feature engineering of credit_card_balance")
    print("****************************************************\n\n")
    
    return credit_df
