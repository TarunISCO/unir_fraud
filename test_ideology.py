import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import utils

warnings.filterwarnings('ignore')

ach_dir = 'data/August/Models/ACH'

model_col = joblib.load('data/August/model_columns.pkl')
print(type(model_col), model_col)
model_col.remove('Sending Method Type_IV')
# model_col.remove('Amount_transformed')

# if True:
#     model_col.remove('Tx count for remitter_transformed')
#     model_col.remove('Tx count for beneficiary_transformed')
#     model_col.remove('Days count from the last tx_transformed')


idology_features = [ 'QUALIFIERS_NEW=Address Does Not Match', 'QUALIFIERS_NEW=Address Longevity Alert',
                     'QUALIFIERS_NEW=Address Stability Alert', 'QUALIFIERS_NEW=Address Velocity Alert',
                     'QUALIFIERS_NEW=Age Above Maximum', 'QUALIFIERS_NEW=Data Strength Alert',
                     'QUALIFIERS_NEW=ITIN Located', 'QUALIFIERS_NEW=Low Risk Score', 'QUALIFIERS_NEW=MOB Not Available',
                     'QUALIFIERS_NEW=No DOB Available', 'QUALIFIERS_NEW=PA DOB Not Available',
                     'QUALIFIERS_NEW=SSN Is Invalid', 'QUALIFIERS_NEW=SSN Not Found',
                     'QUALIFIERS_NEW=Single Address in File', 'QUALIFIERS_NEW=Street Name Does Not Match',
                     'QUALIFIERS_NEW=Street Number Does Not Match', 'QUALIFIERS_NEW=Thin File',
                     'QUALIFIERS_NEW=Warm Address Alert (mail drop)', 'QUALIFIERS_NEW=YOB Does Not Match']

model_col.extend(idology_features)

'''
ACH Transactions
'''

train_file = pd.read_excel('data/Sept/IdologyFeat/Transformed-ACH-1-2017_6-2019.xlsx')

data_train = train_file[model_col]
data_train['Label'] = train_file['Label']
print(f'Train shape = {data_train.shape}')
# print(data_train.isna().sum())


tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9, 10]}]
training_batches = utils.createBatches(data_train, 4)

ach_clf = []
sampling = 6

for batch in training_batches:
    X, y = utils.preprocessunEqualDistribution(batch, sampling)

    rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
    rf_clf.fit(X, y)

    ach_clf.append(rf_clf)


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


"""
Testing

"""


test_file = pd.read_excel('data/Sept/IdologyFeat/Transformed-Aug19ACH.xlsx')
test_file = test_file[test_file['Tx date and time'] > '2019-06-30']

print('Test File', test_file.shape)

ach_x = test_file[model_col]
ach_y = test_file['Label']

print(f'Test x shape = {ach_x.shape}')

ach_pred, ach_prob, ach_y_ = averageTest(ach_x, ach_clf)

conf = confusion_matrix(ach_y, ach_y_)
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions(Cutoff = 0.5)'},
            xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.show()

data_test = test_file.copy()
data_test['y_pred'] = ach_y_
data_test['y_prob'] = ach_prob[:, 1]
data_test['Label'] = ach_y

test_ach = data_test[data_test['Sending Method Type'].isin(['BANK ACCOUNT'])]
test_ach['y_pred_60'] = (test_ach['y_prob'] > 0.60).astype(int)
test_ach['y_pred_40'] = (test_ach['y_prob'] > 0.40).astype(int)


conf = confusion_matrix(test_ach['Label'], test_ach['y_pred_40'])
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions(Cutoff = 0.4)'},
            xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.show()


conf = confusion_matrix(test_ach['Label'], test_ach['y_pred_60'])
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions(Cutoff = 0.6)'},
            xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.show()


print('Cutoff = 0.40')

tp = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred_40'] == 1)]
fp = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred_40'] == 1)]
tn = test_ach[(test_ach['Label'] == 0) & (test_ach['y_pred_40'] == 0)]
fn = test_ach[(test_ach['Label'] == 1) & (test_ach['y_pred_40'] == 0)]

print(f'True Positive : {tp.shape} transactions and ${tp["Amount"].sum()}')
print(f'True Negative : {tn.shape} transactions and ${tn["Amount"].sum()}')
print(f'False Positive : {fp.shape} transactions and ${fp["Amount"].sum()}')
print(f'False Negative : {fn.shape} transactions and ${fn["Amount"].sum()}')


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
