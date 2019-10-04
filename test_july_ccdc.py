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

model_col = joblib.load('data/August/model_columns.pkl')
print(model_col)
# No need to use Sending Method Type_IV with separate models
# model_col.remove('Sending Method Type_IV')
model_col.remove('Amount_transformed')

# if True:
#     model_col.remove('Tx count for remitter_transformed')
#     model_col.remove('Tx count for beneficiary_transformed')
#     model_col.remove('Days count from the last tx_transformed')

if True:
    model_col.remove('User Level_IV')


'''
CCDC Transactions
'''

train_file = pd.read_excel('data/Sept/Transformed-CCDC-1-2017_6-2019.xlsx')

data_train = train_file[model_col]
data_train['Label'] = train_file['Label']

test_file = pd.read_excel('data/Sept/Transformed-Aug19CCDC.xlsx')

print('Test data', test_file.shape)

tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9, 10]}]
training_batches = utils.createBatches(data_train, 4)

ccdc_clf = []
sampling = 7

for batch in training_batches:
    X, y = utils.preprocessunEqualDistribution(batch, sampling)

    rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
    rf_clf.fit(X, y)

    ccdc_clf.append(rf_clf)


def averageTest(X_test, clfs):
    """
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


ccdc_x = test_file[model_col]
ccdc_y = test_file['Label']

ccdc_pred, ccdc_prob, ccdc_y_ = averageTest(ccdc_x, ccdc_clf)

conf = confusion_matrix(ccdc_y, ccdc_y_)
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()

data_test = test_file.copy()
data_test['y_pred'] = ccdc_y_
data_test['y_prob'] = ccdc_prob[:, 1]
# data_test.to_csv('observations/july/ach_test.csv')

test_ccdc = data_test
test_ccdc['y_pred_35'] = (test_ccdc['y_prob'] > 0.35).astype(int)
test_ccdc['y_pred_40'] = (test_ccdc['y_prob'] > 0.40).astype(int)

# conf = confusion_matrix(test_ach['Label'], test_ach['y_pred'])
# sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
# plt.show()

print('Cutoff = 0.35')

tp = test_ccdc[(test_ccdc['Label'] == 1) & (test_ccdc['y_pred_35'] == 1)]
fp = test_ccdc[(test_ccdc['Label'] == 0) & (test_ccdc['y_pred_35'] == 1)]
tn = test_ccdc[(test_ccdc['Label'] == 0) & (test_ccdc['y_pred_35'] == 0)]
fn = test_ccdc[(test_ccdc['Label'] == 1) & (test_ccdc['y_pred_35'] == 0)]

print(f'True Positive : {tp.shape} transactions and ${tp["Amount"].sum()}')
print(f'True Negative : {tn.shape} transactions and ${tn["Amount"].sum()}')
print(f'False Positive : {fp.shape} transactions and ${fp["Amount"].sum()}')
print(f'False Negative : {fn.shape} transactions and ${fn["Amount"].sum()}')

print('Cutoff = 0.40')

tp = test_ccdc[(test_ccdc['Label'] == 1) & (test_ccdc['y_pred_40'] == 1)]
fp = test_ccdc[(test_ccdc['Label'] == 0) & (test_ccdc['y_pred_40'] == 1)]
tn = test_ccdc[(test_ccdc['Label'] == 0) & (test_ccdc['y_pred_40'] == 0)]
fn = test_ccdc[(test_ccdc['Label'] == 1) & (test_ccdc['y_pred_40'] == 0)]

print(f'True Positive : {tp.shape} transactions and ${tp["Amount"].sum()}')
print(f'True Negative : {tn.shape} transactions and ${tn["Amount"].sum()}')
print(f'False Positive : {fp.shape} transactions and ${fp["Amount"].sum()}')
print(f'False Negative : {fn.shape} transactions and ${fn["Amount"].sum()}')

# conf = confusion_matrix(test_ach['Label'], test_ach['y_pred_35'])
# sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
# plt.show()

print('Cutoff = 0.5')

tp = test_ccdc[(test_ccdc['Label'] == 1) & (test_ccdc['y_pred'] == 1)]
fp = test_ccdc[(test_ccdc['Label'] == 0) & (test_ccdc['y_pred'] == 1)]
tn = test_ccdc[(test_ccdc['Label'] == 0) & (test_ccdc['y_pred'] == 0)]
fn = test_ccdc[(test_ccdc['Label'] == 1) & (test_ccdc['y_pred'] == 0)]

print(f'True Positive : {tp.shape} transactions and ${tp["Amount"].sum()}')
print(f'True Negative : {tn.shape} transactions and ${tn["Amount"].sum()}')
print(f'False Positive : {fp.shape} transactions and ${fp["Amount"].sum()}')
print(f'False Negative : {fn.shape} transactions and ${fn["Amount"].sum()}')
