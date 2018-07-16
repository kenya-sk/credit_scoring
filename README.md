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
&emsp;│&emsp;├ out_mpl.npy  
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
