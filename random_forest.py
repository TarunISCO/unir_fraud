import pandas as pd
import pydotplus as pydotplus
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from sklearn import tree
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('data/NewFeaturesRecent.csv')
data = data[data['Sending Method Type'] == 'BANK ACCOUNT']

data.drop(['Sending Method Type_IV', 'FINAL_RESULT_EMAILAGE=FAIL', 'Tx count for remitter_IV',
           'IP_RISK_LEVEL_ID=1.0', 'IP_RISK_LEVEL_ID=2.0', 'IP_RISK_LEVEL_ID=3.0', 'IP_RISK_LEVEL_ID=4.0',
           'IP_RISK_LEVEL_ID=5.0', 'IP_RISK_LEVEL_ID=6.0', 'FINAL_RESULT_EMAILAGE=PASS',
           'FINAL_RESULT_EMAILAGE=SOFT FAIL', 'IP_USER_TYPE_IV', 'IS_First_Tx for Beneficiary'], axis=1, inplace=True)

data_test = data[(data['Tx date and time'] > '2018-09-30') & (data['Tx date and time'] <= '2018-11-30')]
data_train = data[(data['Tx date and time'] <= '2018-09-30') & (data['Tx date and time'] > '2017-11-30')]

# print(data_train.shape, data_test.shape)

data_train = data_train.drop(['Tx date and time', 'Sending Method Type', 'Amount'], axis=1)

X_train = data_train.drop(['Label'], axis=1)
y_train = data_train['Label']
# X_train = data_train[['Tx count for remitter', 'Tx count for beneficiary', 'Days count from the last tx']]

X_test = data_test.drop(['Label', 'Tx date and time', 'Sending Method Type', 'Amount'], axis=1)
# X_test = data_test[['Tx count for remitter', 'Tx count for beneficiary', 'Days count from the last tx']]
y_test = data_test['Label']

# print(data_train.shape, data_test.shape)


# print(data_test['Tx date and time'].value_counts())


# classifier = RandomForestClassifier(random_state=101)
# classifier.fit(X_train, y_train)

# i_tree = 0
# for tree_in_forest in classifier.estimators_:
#     with open('observations/tree/tree_' + str(i_tree) + '.dot', 'w') as my_file:
#         tree.export_graphviz(tree_in_forest, out_file=my_file,
#                              feature_names=X_train.columns.values,
#                              class_names=['0', '1'],
#                              rounded=True, proportion=False,
#                              precision=2, filled=True)
#     i_tree = i_tree + 1

# y_pred = classifier.predict(X_test)
#
# conf = confusion_matrix(y_test, y_pred)
#
# sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
# plt.show()


# feats = {} # a dict to hold feature_name: feature_importance
# for feature, importance in zip(X_train.columns, classifier.feature_importances_):
#     feats[feature] = importance
#
# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
# print(importances.sort_values(by='importance'))

clfs = []

sampling = 6
tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9, 10]}]
ind = 1

training_batches = utils.createBatches(data_train, 4)
for batch in training_batches:
    # X = batch.drop('Label', axis=1)
    # y = batch['Label']
    X, y = utils.preprocessunEqualDistribution(batch, sampling)

    rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
    # rf_clf = RandomForestClassifier(random_state=100)
    rf_clf.fit(X, y)
    clfs.append(rf_clf)

    # feats = {} # a dict to hold feature_name: feature_importance
    # for feature, importance in zip(X_train.columns, rf_clf.feature_importances_):
    #     feats[feature] = importance
    #
    # importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
    # print(ind)
    # print(importances.sort_values(by='importance'))
    ind = ind + 1


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

label_df = pd.DataFrame(data=dict(y_test=y_test,
                                  y_pred=y_pred_,
                                  y_prob=y_prob[:, 1],
                                  amount=data_test['Amount'], ))


def amount_statistics_account(df, cutoff):
    print('\n\n\n################################################################################################')
    print('\n\n\nAmount statistics for Bank account transactions with cutoff = ', cutoff)
    total = df['amount'].sum()
    print('Total amount in Bank transactions', total)

    tp = df[(df['y_test'] == 1) & (df['y_prob'] >= cutoff)]['amount']
    fp = df[(df['y_test'] == 0) & (df['y_prob'] >= cutoff)]['amount']
    fn = df[(df['y_test'] == 1) & (df['y_prob'] < cutoff)]['amount']
    tn = df[(df['y_test'] == 0) & (df['y_prob'] < cutoff)]['amount']

    print('\n\nFraud amount saved by the model\n', tp.shape)
    print('total amount, percentage = ', tp.sum(), tp.sum() * 100 / total)

    print('\n\nGenuine amount dealt by the model\n', tn.shape)
    print('total amount, percentage = ', tn.sum(), tn.sum() * 100 / total)

    print('\n\nFraud tx not detected\n', fn.shape)
    print('total amount, percentage = ', fn.sum(), fn.sum() * 100 / total)

    print('\n\nGenuine tx wrongly detected\n', fp.shape)
    print('total amount, percentage = ', fp.sum(), fp.sum() * 100 / total)


print('Features used for training')
for col in X_train.columns:
    print(col)

amount_statistics_account(label_df, 0.3)
amount_statistics_account(label_df, 0.4)
amount_statistics_account(label_df, 0.5)
amount_statistics_account(label_df, 0.6)
amount_statistics_account(label_df, 0.7)
