import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import utils

warnings.filterwarnings('ignore')

"""
Train single model on single file after selecting the features and sampling
"""
# TODO: Enable/Disable storing models as per requirements
# TODO: Change 'model_path' and 'model_name' for ACH/CCDC accordingly

data_train = pd.read_excel('data/August/20/Transformed-Combined-1-2017_5-2019.xlsx')

data_ = pd.read_excel('data/August/20/Transformed-Jul19Combined.xlsx')

# data_ = data_[data_['Sending Method Type'].isin(['BANK ACCOUNT'])]

print(data_.shape)

features = ['Age_transformed', 'Amount_transformed', 'IP_RISK_LEVEL_ID=4.0', 'FINAL_RESULT_EMAILAGE=FAIL',
            'FINAL_RESULT_EMAILAGE=SOFT FAIL', 'IP_RISK_LEVEL_ID=1.0', 'IP_RISK_LEVEL_ID=3.0',
            'IP_RISK_LEVEL_ID=2.0', 'Payer_IV', 'Beneficiary State_IV', 'IP_RISK_LEVEL_ID=6.0', 'Origin State_IV',
            'FINAL_RESULT_EMAILAGE=PASS', 'DOMAIN_NAME_IV', 'IP_RISK_LEVEL_ID=5.0', 'IP_USER_TYPE_IV',
            'Joining Time_transformed', 'Label', 'Tx date and time', 'Amount', 'Sending Method Type', 'Sending Method Type_IV']

# if True:
#     features.append('Tx count for remitter_transformed')
#     features.append('Tx count for beneficiary_transformed')
#     features.append('Days count from the last tx_transformed')
#
# if True:
#     features.append('User Level_IV')

data_test = data_[features]

# data_test = data[(data['Tx date and time'] >= '2019-06-01')]
# data_train = data[(data['Tx date and time'] >= '2017-01-01')]
# data_train = data[(data['Tx date and time'] >= '2017-01-01') & (data['Tx date and time'] < '2019-06-01')]
# data_train = data_train[(data_train['Tx date and time'] >= '2017-01-01') & (data_train['Tx date and time'] < '2019-01-01')]

print('train and test data shape')
print(data_train.shape, data_test.shape)

data_train = data_train[features]
data_train = data_train.drop(['Tx date and time', 'Amount', 'Sending Method Type'], axis=1)

X_test = data_test.drop(['Label', 'Tx date and time', 'Amount', 'Sending Method Type'], axis=1)
y_test = data_test['Label']

tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9, 10]}]
training_batches = utils.createBatches(data_train, 4)

clfs = []
sampling = 6
ind = 1

model_path = 'observations/july/models/ACH'

for batch in training_batches:
    X, y = utils.preprocessunEqualDistribution(batch, sampling)

    rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
    rf_clf.fit(X, y)
    clfs.append(rf_clf)
    model_name = f'RF_ACH_{ind}'
    ind = ind + 1
    # joblib.dump(rf_clf, os.path.join(model_path, model_name))


def averageTest(X_test):
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


y_pred, y_prob, y_pred_ = averageTest(X_test)
conf = confusion_matrix(y_test, y_pred_)
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()


data_['y_pred_5'] = y_pred_
data_['y_prob'] = y_prob[:, 1]
# data_.to_csv('data/August/20/Transformed-Jul19Combined_scores.csv')

data_test['y_pred'] = y_pred_
data_test['y_prob'] = y_prob[:, 1]
# data_test.to_csv('observations/july/ach_test.csv')



