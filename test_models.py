import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import utils

warnings.filterwarnings('ignore')


test_file = pd.read_excel('data/Sept/Transformed-Aug19ACH.xlsx')

model_col = joblib.load('data/August/model_columns.pkl')
print(model_col)
model_col.remove('Sending Method Type_IV')


test_file = test_file[test_file['Tx date and time'] > '2019-06-30']


clf_path = 'data/july/models/ACH/'
clfs = []


for file in os.listdir(clf_path):
    clf = joblib.load(os.path.join(clf_path, file))
    clfs.append(clf)


def averageTest(X_test, clfs):
    """
    :param clfs:
    :type X_test: dataframe
    """
    y_pred = [0] * X_test.shape[0]
    y_prob = [[0] * 2 for i in range(X_test.shape[0])]

    for clf in clfs:
        y_pred = y_pred + clf.predict(X_test)
        y_prob = y_prob + clf.predict_proba(X_test)

    y_pred = (y_pred > 2).astype(int)
    y_prob = y_prob / 4
    y_pred_ = (y_prob[:, 1] > 0.5).astype(int)
    return y_pred, y_prob, y_pred_


ach_x = test_file[model_col]
ach_y = test_file['Label']

ach_pred, ach_prob, ach_y_ = averageTest(ach_x, clfs)

conf = confusion_matrix(ach_y, ach_y_)
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions(Cutoff = 0.5)'})
plt.show()

data_test = test_file.copy()
data_test['y_pred'] = ach_y_
data_test['y_prob'] = ach_prob[:, 1]
data_test['Label'] = ach_y

csv_df = data_test.drop(['Age', 'Age_transformed', 'Amount_transformed', 'Beneficiary State_IV', 'DOMAIN_NAME_IV',
                         'Days count from the last tx_transformed', 'FINAL_RESULT_EMAILAGE=FAIL',
                         'FINAL_RESULT_EMAILAGE=PASS', 'FINAL_RESULT_EMAILAGE=SOFT FAIL', 'IP_RISK_LEVEL_ID=1.0',
                         'IP_RISK_LEVEL_ID=2.0', 'IP_RISK_LEVEL_ID=3.0', 'IP_RISK_LEVEL_ID=4.0', 'IP_RISK_LEVEL_ID=5.0',
                         'IP_RISK_LEVEL_ID=6.0', 'IP_USER_TYPE_IV', 'Joining Time', 'Joining Time_transformed', 'Label',
                         'Origin State_IV', 'Payer_IV', 'Sending Method Type_IV', 'Tx count for beneficiary_transformed']
                        , axis=1)
# data_test.to_csv('observations/july/ach_test.csv')
# csv_df.to_csv('data/Sept/testresults_july-aug2019_cutoff=7.csv', index=False)

test_ach = data_test[data_test['Sending Method Type'].isin(['BANK ACCOUNT'])]
test_ach['y_pred_60'] = (test_ach['y_prob'] > 0.60).astype(int)
test_ach['y_pred_40'] = (test_ach['y_prob'] > 0.40).astype(int)

# conf = confusion_matrix(test_ach['Label'], test_ach['y_pred'])
# sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
# plt.show()


print('Cutoff = 0.40')

tp = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred_40'] == 1)]
fp = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred_40'] == 1)]
tn = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred_40'] == 0)]
fn = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred_40'] == 0)]

print(f'True Positive : {tp.shape} transactions and ${tp["Amount"].sum()}')
print(f'True Negative : {tn.shape} transactions and ${tn["Amount"].sum()}')
print(f'False Positive : {fp.shape} transactions and ${fp["Amount"].sum()}')
print(f'False Negative : {fn.shape} transactions and ${fn["Amount"].sum()}')

conf = confusion_matrix(test_ach['Label'], test_ach['y_pred_40'])
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions(Cutoff = 0.4)'})
plt.show()


conf = confusion_matrix(test_ach['Label'], test_ach['y_pred_60'])
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions(Cutoff = 0.6)'})
plt.show()

print('Cutoff = 0.5')

tp = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred'] == 1)]
fp = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred'] == 1)]
tn = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred'] == 0)]
fn = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred'] == 0)]

print(f'True Positive : {tp.shape} transactions and ${tp["Amount"].sum()}')
print(f'True Negative : {tn.shape} transactions and ${tn["Amount"].sum()}')
print(f'False Positive : {fp.shape} transactions and ${fp["Amount"].sum()}')
print(f'False Negative : {fn.shape} transactions and ${fn["Amount"].sum()}')


print('Cutoff = 0.60')

tp = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred_60'] == 1)]
fp = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred_60'] == 1)]
tn = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred_60'] == 0)]
fn = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred_60'] == 0)]

print(f'True Positive : {tp.shape} transactions and ${tp["Amount"].sum()}')
print(f'True Negative : {tn.shape} transactions and ${tn["Amount"].sum()}')
print(f'False Positive : {fp.shape} transactions and ${fp["Amount"].sum()}')
print(f'False Negative : {fn.shape} transactions and ${fn["Amount"].sum()}')