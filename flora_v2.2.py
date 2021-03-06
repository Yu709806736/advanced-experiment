import operator
import sys

import numpy as np
from imblearn.over_sampling import SMOTE
import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from DI import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

DEFAULT_MAX_WINDOW_SIZE: int = 5000
DEFAULT_MIN_WINDOW_SIZE: int = 2000
DEFAULT_INIT_WINDOW_SIZE: int = 500

df = pd.read_csv("D:\\SUSTech\\2019 fall\\innovation1\\credit-card-fraud-detection\\creditcard.csv")
# print(df.head(10))
# df = df.head(20000)

# description
# print(df.describe())

# check NULL values
# print(df.isnull().sum().max())

# get column names
# print(df.columns)

# check ratio b|w frauds and non frauds
# print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100, 2), '% of the dataset')   99.83%
# print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100, 2), '% of the dataset')   0.17%

lc: float = 0  # lower threshold of N/S
hc: float = 0.17 / 99.83 * 3  # higher threshold of N/S
pacc: float = 0.99  # threshold of predict accuracy
prec: float
# print(lc, hc)

# x = df.drop('Class', axis=1).drop('Time', axis=1).values.tolist()
x = df.drop('Class', axis=1).drop('Amount', axis=1).values.tolist()
y = df['Class'].values.tolist()
# count = 0   check the number of frauds
# for i in range(len(y)):
#     if y[i] == 1:
#         print(i)
#         count += 1
# print(count)

dt = DecisionTreeClassifier()
sm = SMOTE(sampling_strategy=0.05, k_neighbors=10)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
train_size: int = int(0.8 * len(x))
test_size: int = len(x) - train_size
x_train = x[:train_size]
x_test = x[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
# SMOTE resampling and sort by time
x_train, y_train = sm.fit_resample(x_train, y_train)
entries = list(zip(x_train, y_train))
entries.sort(key=lambda in_: in_[0])
x_train,  y_train = zip(*entries)
x_train = list(x_train)
y_train = list(y_train)
for i in x_train:
    i.pop(0)
for i in x_test:
    i.pop(0)
# x_train = x_train[:, 1:]
# x_test = x_test[:, 1:]
train_size = len(x_train)

window_size = DEFAULT_INIT_WINDOW_SIZE  # initialize the window size
min_pos_por = 0.2  # minimum positive classified proportion
min_neg_por = 0.5  # minimum negative classified proportion
ADES = ADes()
PDES = PDes()
NDES = NDes()
window = [[], []]
accuracy: float = 0


def learn(instance):
    global ADES, PDES, NDES
    if instance[1] == 1:
        match: bool = False
        for i in range(len(ADES.items)):
            if operator.eq(instance[0], ADES.items[i][0]) and instance[1] == ADES.items[i][1]:
                ADES.ap[i] += 1
                match = True
                break
        for i in range(len(PDES.items)):
            if operator.eq(instance[0], PDES.items[i][0]) and instance[1] == PDES.items[i][1]:
                PDES.ppos[i] += 1
                break
        for i in range(len(NDES.items)):
            if operator.eq(instance[0], NDES.items[i][0]):
                item = NDES.items.pop(i)
                nn = NDES.nn.pop(i)
                PDES.items.append(item)
                PDES.ppos.append(1)
                PDES.pneg.append(nn)
                break
        if not match:
            ADES.items.append(instance)
            ADES.ap.append(1)
    else:
        match: bool = False
        for i in range(len(NDES.items)):
            if operator.eq(instance[0], NDES.items[i][0]) and instance[1] == NDES.items[i][1]:
                NDES.nn[i] += 1
                match = True
                break
        for i in range(len(PDES.items)):
            if operator.eq(instance[0], PDES.items[i][0]) and instance[1] == PDES.items[i][1]:
                PDES.pneg[i] += 1
                break
        for i in range(len(ADES.items)):
            if operator.eq(instance[0], ADES.items[i][0]):
                item = ADES.items.pop(i)
                ap = ADES.ap.pop(i)
                PDES.items.append(item)
                PDES.ppos.append(1)
                PDES.pneg.append(ap)
                break
        if not match:
            NDES.items.append(instance)
            NDES.nn.append(1)


def forget(instance):
    global ADES, PDES, NDES
    if instance[1] == 1:
        for i in range(len(ADES.items)):
            # if instance == ADES.items[i]:
            if operator.eq(instance[0], ADES.items[i][0]) and instance[1] == ADES.items[i][1]:
                ADES.ap[i] -= 1
                if ADES.ap[i] == 0:
                    ADES.ap.pop(i)
                    ADES.items.pop(i)
                break
        for i in range(len(PDES.items)):
            # if instance == PDES.items[i]:
            if operator.eq(instance[0], PDES.items[i][0]) and instance[1] == PDES.items[i][1]:
                PDES.ppos[i] -= 1
                if PDES.ppos[i] == 0:
                    pn = PDES.pneg.pop(i)
                    item = PDES.items.pop(i)
                    PDES.ppos.pop(i)
                    NDES.items.append(item)
                    NDES.nn.append(pn)
                break
    else:
        for i in range(len(NDES.items)):
            # if instance == NDES.items[i]:
            # print('instance =', instance[0])
            # print('NDES.items[i] =', NDES.items[i][0])
            if operator.eq(instance[0], NDES.items[i][0]) and instance[1] == NDES.items[i][1]:
                # print("TRUE")
                NDES.nn[i] -= 1
                if NDES.nn[i] == 0:
                    NDES.nn.pop(i)
                    NDES.items.pop(i)
                break
        for i in range(len(PDES.items)):
            # if instance == PDES.items[i]:
            if operator.eq(instance[0], PDES.items[i][0]) and instance[1] == PDES.items[i][1]:
                PDES.ppos[i] -= 1
                if PDES.ppos[i] == 0:
                    pn = PDES.pneg.pop(i)
                    item = PDES.items.pop(i)
                    PDES.ppos.pop(i)
                    NDES.items.append(item)
                    NDES.nn.append(pn)
                break


def adjust_window():
    global ADES, PDES, NDES
    n: int = ADES.getN()
    s: int = len(ADES.items)
    if s == 0:
        return
    ratio: float = float(n) / float(s)
    global window_size
    global recall, TN, FN, TP, FP, pacc, prec
    if window_size < DEFAULT_MIN_WINDOW_SIZE:
        return
    if window_size > DEFAULT_MAX_WINDOW_SIZE:
        if len(ADES.items) < min_pos_por * window_size:
            i: int = 0
            pointer: int = 0
            while i < 1:
                if window[1][pointer] == 1:
                    pointer += 1
                    continue
                _inst = [window[0].pop(pointer), window[1].pop(pointer)]
                forget(_inst)
                i += 1
        elif len(NDES.items) < min_neg_por * window_size:
            i: int = 0
            pointer: int = 0
            while i < 1:
                if window[1][pointer] == 0:
                    pointer += 1
                    continue
                _inst = [window[0].pop(pointer), window[1].pop(pointer)]
                forget(_inst)
                i += 1
        return
    if TP + FN < 20:
        prec = -1
    else:
        prec = 0.83

    metric: bool = False
    if prec != -1:
        if recall > prec:
            metric = True
    elif accuracy > pacc:
        metric = True

    if ratio < lc or accuracy < pacc or recall < prec:  # and (len(PDES.items) != 0):
        remove = int(window_size * 0.2)
        window_size -= remove
        pos: int = 0
        for i in range(remove):
            if len(ADES.items) <= min_pos_por * window_size and window[1][pos] == 1:
                pos += 1
                continue
            elif len(NDES.items) <= min_neg_por * window_size and window[1][pos] == 0:
                pos += 1
                continue
            _inst = [window[0].pop(pos), window[1].pop(pos)]
            forget(_inst)
        # print(jj, "situ 1:", window_size, file=sys.stderr)
        return
    elif (ratio > (2 * hc) and metric) or window_size > DEFAULT_MAX_WINDOW_SIZE:
        window_size -= 2
        i: int = 0
        pointer: int = 0
        while i < 2:
            if len(ADES.items) <= min_pos_por * window_size and window[1][pointer] == 1:
                pointer += 1
                continue
            elif len(NDES.items) <= min_neg_por * window_size and window[1][pointer] == 0:
                pointer += 1
                continue
            _inst = [window[0].pop(pointer), window[1].pop(pointer)]
            forget(_inst)
            i += 1
        # _inst = [window[0].pop(0), window[1].pop(0)]
        # forget(_inst)
        # _inst = [window[0].pop(0), window[1].pop(0)]
        # forget(_inst)
        # print(jj, "situ 2:", window_size, file=sys.stderr)
        return
    elif ratio > hc and metric:
        window_size -= 1
        i: int = 0
        pointer: int = 0
        while i < 1:
            if len(ADES.items) <= min_pos_por * window_size and window[1][pointer] == 1:
                pointer += 1
                continue
            elif len(NDES.items) <= min_neg_por * window_size and window[1][pointer] == 0:
                pointer += 1
                continue
            _inst = [window[0].pop(pointer), window[1].pop(pointer)]
            forget(_inst)
            i += 1
        # print(jj, "situ 3:", window_size, file=sys.stderr)
        return
    # print(jj, "situ 4:", window_size, file=sys.stderr)


for j in range(window_size):
    window[0].append(x_train[j])
    window[1].append(y_train[j])
    learn([x_train[j], y_train[j]])

TP: int = 0
FN: int = 0
TN: int = 0
FP: int = 0
for j in range(DEFAULT_INIT_WINDOW_SIZE, train_size):
    dt = DecisionTreeClassifier()
    dt.fit(window[0], window[1])
    score = dt.score(np.array(x_train[j]).reshape(1, -1), np.array(y_train[j]).reshape(1, -1))
    window_size += 1
    window[0].append(x_train[j])
    window[1].append(y_train[j])
    inst = [x_train[j], y_train[j]]
    learn(inst)
    # if y_train[j] == 1:
    #     window[0].append(x_train[j])
    #     window[1].append(y_train[j])
    #     inst = [x_train[j], y_train[j]]
    #     learn(inst)
    if score == 1 and y_train[j] == 1:
        TP += 1
        # print("TP =", TP)
    elif score == 0 and y_train[j] == 1:
        FN += 1
        # print("FN =", FN)
    elif score == 1 and y_train[j] == 0:
        TN += 1
        # print("TN =", TN)
    elif score == 0 and y_train[j] == 0:
        FP += 1
        # print("FP =", FP)
    accuracy = (accuracy * (j - DEFAULT_INIT_WINDOW_SIZE) + score) / (j - DEFAULT_INIT_WINDOW_SIZE + 1)
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
    adjust_window()
    print("j =", j, "window size =", window_size, 'accuracy =', accuracy, 'recall =', recall)
    if len(PDES.items) > 0:
        print("len(ADES) =", len(ADES.items), "len(PDES) =", len(PDES.items),
              "len(NDES) =", len(NDES.items), file=sys.stderr)
        pass
    else:
        print("len(ADES) =", len(ADES.items), "len(PDES) =", len(PDES.items), "len(NDES) =", len(NDES.items))
        pass

dt_pred = dt.predict(x_test)
confu_matrix = confusion_matrix(y_test, dt_pred)
FPR, TPR, thresholds = roc_curve(y_test, dt_pred)
ras = roc_auc_score(y_test, dt_pred)
fig, ax = plt.subplots()
plt.title("ROC Curve of testing set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(FPR, TPR, label='Decition Tree Classifier Score: {:.4f}'.format(ras))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
             arrowprops=dict(facecolor='#6E726D', shrink=0.05), )
plt.legend()
plt.savefig('./test5.jpg')
plt.show()

print("Confusion matrix:")
print(confu_matrix)
print("Report:")
print(classification_report(y_test, dt_pred))
print("AUC score of ROC curve:", ras)
