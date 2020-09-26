DEFAULT_MAX_WINDOW_SIZE: int = 5000  
DEFAULT_MIN_WINDOW_SIZE: int = 2000  
DEFAULT_INIT_WINDOW_SIZE: int = 1000  

# Whole data set  

lc: float = 0  # lower coverage threshold of N/S  
hc: float = 0.17 / 99.83 * 3  # higher coverage threshold of N/S  
pacc: float = 0.99  # threshold of predict accuracy  
prec: float = 0.85  # threshold of predict recall  

Training set: size -- 227844  
&nbsp;&nbsp;  window size = 2079  
&nbsp;&nbsp;  accuracy = 0.9842227071348277  
&nbsp;&nbsp;  recall = 0.8313253012048193  
  
TP = 345, FN = 70, TN = 222921, FP = 3509  

Testing set: size -- 56962  
&nbsp;&nbsp;  Confusion matrix:  
&nbsp;&nbsp;&nbsp;&nbsp;    [[55347  1540]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     [   13    62]]  
     
&nbsp;&nbsp;  Report:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                  precision    recall  f1-score   support  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    0       1.00      0.97      0.99     56887  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    1       0.04      0.83      0.07        75  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  accuracy                           0.97     56962  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   macro avg       0.52      0.90      0.53     56962  
&nbsp;&nbsp;&nbsp;&nbsp;  weighted avg       1.00      0.97      0.98     56962  
    
&nbsp;&nbsp;  AUC score of ROC curve: 0.8997977276589262  
&nbsp;&nbsp;  ROC curve stored in: ![dt_roc.jpg](https://github.com/Yu709806736/advanced-experiment/blob/master/images/dt_roc.jpg)  
