import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import warnings
import seaborn as sns

import utils

plt.interactive(False)

warnings.filterwarnings('ignore')

########################################################################################
# Import Data
########################################################################################

data = pd.read_csv('data/NewFeatures.csv')

data = data[['Tx count for remitter', 'Tx count for beneficiary', 'Days count from the last tx', 'Amount',
             'Label', 'Sending Method Type', 'Tx date and time', 'IS_First_Tx for Beneficiary',
             'Sending Method Type_IV']]

data_test = data[(data['Tx date and time'] > '2018-09-30') & (data['Tx date and time'] <= '2018-11-30')]
data_train = data[(data['Tx date and time'] <= '2018-09-30') & (data['Tx date and time'] > '2016-12-31')]

data_train = data_train.drop(['Tx date and time', 'Sending Method Type', 'Tx count for remitter',
                              'IS_First_Tx for Beneficiary', 'Days count from the last tx'], axis=1)

X_test = data_test.drop(['Label', 'Tx date and time', 'Sending Method Type', 'Tx count for remitter',
                         'IS_First_Tx for Beneficiary', 'Days count from the last tx'], axis=1)
y_test = data_test['Label']

clfs = []

sampling = 5
tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9]}]

training_batches = utils.createBatches(data_train, 4)
ind = 1
for batch in training_batches:
    # X = batch.drop('Label', axis=1)
    # y = batch['Label']
    X, y = utils.preprocessunEqualDistribution(batch, sampling)

    rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
    rf_clf.fit(X, y)
    print('\n Random Forest', ind)
    print(rf_clf.best_params_)
    print(rf_clf.best_estimator_.feature_importances_)
    clfs.append(rf_clf)
    filename = 'data/models/batches/model_' + str(sampling * 10) + '-' + str((10 - sampling) * 10) + '_' + str(ind)
    ind = ind + 1
    # pickle.dump(rf_clf, open(filename, 'wb'))


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
# print(y_pred_)
conf = confusion_matrix(y_test, y_pred_)
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
# conf_img_name = 'observations/distribution_new/' + str(sampling * 10) + '-' + str(
#     (10 - sampling) * 10) + '_conf_matrix.png'
# plt.savefig(conf_img_name)
plt.show()

# print(y_pred.shape, y_prob.shape)
label_df = pd.DataFrame(data=dict(y_test=y_test,
                                  y_pred=y_pred_,
                                  y_prob=y_prob[:, 1],
                                  amount=data_test['Amount'],
                                  sending_method=data_test['Sending Method Type'],
                                  rem_count=data_test['Tx count for remitter'],
                                  ben_count=data_test['Tx count for beneficiary'],
                                  days_from_last_txn=data_test['Days count from the last tx']))

true_positive = label_df[(label_df['y_test'] == 1) & (label_df['y_pred'] == 1)]
false_positive = label_df[(label_df['y_test'] == 0) & (label_df['y_pred'] == 1)]
false_negative = label_df[(label_df['y_test'] == 1) & (label_df['y_pred'] == 0)]
true_negative = label_df[(label_df['y_test'] == 0) & (label_df['y_pred'] == 0)]

# true_positive.to_csv('observations/csv/tp.csv')
# false_positive.to_csv('observations/csv/fp.csv')
# true_negative.to_csv('observations/csv/tn.csv')
# false_negative.to_csv('observations/csv/fn.csv')


def amount_description(start_prob, end_prob):

    fraud_saved_by_model = true_positive[true_positive['y_prob'] > end_prob]['amount']

    fraud_saved_by_third_party = true_positive[true_positive['y_prob'] <= end_prob]['amount'].append(
        false_negative[false_negative['y_prob'] > start_prob]['amount'], ignore_index=True)

    print(true_positive[true_positive['y_prob'] <= end_prob]['amount'].shape)
    print(false_negative[false_negative['y_prob'] > start_prob]['amount'].shape)

    genuine_dealt_by_model = true_negative[true_negative['y_prob'] <= start_prob]['amount']

    genuine_dealt_by_third_party = true_negative[true_negative['y_prob'] > start_prob]['amount'].append(
        false_positive[false_positive['y_prob'] < end_prob]['amount'], ignore_index=True)

    fraud_amount_escaped = false_negative[false_negative['y_prob'] <= start_prob]['amount']

    genuine_amount_troubled = false_positive[false_positive['y_prob'] > end_prob]['amount']

    fn_bank_account = false_negative[(false_negative['y_prob'] > start_prob) &
                                     (false_negative['sending_method'].isin(['BANK ACCOUNT']))]

    fp_bank_account = false_positive[
        (false_positive['y_prob'] < end_prob) & (false_positive['sending_method'].isin(['BANK ACCOUNT']))]

    total_amount = data_test['Amount'].sum()

    print('total amount', total_amount)

    print('\n\nFraud amount saved by the model\n', fraud_saved_by_model.describe())
    print('total amount, percentage = ', fraud_saved_by_model.sum(), fraud_saved_by_model.sum() * 100 / total_amount)

    print('\n\nFraud amount saved by the third party\n', fraud_saved_by_third_party.describe())
    print('total amount, percentage = ', fraud_saved_by_third_party.sum(),
          fraud_saved_by_third_party.sum() * 100 / total_amount)

    print('\n\nGenuine amount dealt by the model\n', genuine_dealt_by_model.describe())
    print('total amount, percentage = ', genuine_dealt_by_model.sum(),
          genuine_dealt_by_model.sum() * 100 / total_amount)

    print('\n\nGenuine amount dealt by the third party\n', genuine_dealt_by_third_party.describe())
    print('total amount, percentage = ', genuine_dealt_by_third_party.sum(),
          genuine_dealt_by_third_party.sum() * 100 / total_amount)

    print('\n\nFraud amount escaped\n', fraud_amount_escaped.describe())
    print('total amount, percentage = ', fraud_amount_escaped.sum(),
          fraud_amount_escaped.sum() * 100 / total_amount)

    print('\n\nGenuine amount troubled\n', genuine_amount_troubled.describe())
    print('total amount, percentage = ', genuine_amount_troubled.sum(),
          genuine_amount_troubled.sum() * 100 / total_amount)

    print('\n\nFalse Negative with Sending Method = Bank Account', fn_bank_account.shape[0],
          fn_bank_account['amount'].sum())

    print('\n\nFalse Positive with Sending Method = Bank Account', fp_bank_account.shape[0],
          fp_bank_account['amount'].sum())


# amount_description(0.3, 0.7)
# amount_description(0.4, 0.6)

print('Rem Count')
print('True Positive\n', true_positive['rem_count'].value_counts())
print('False Negative\n', false_negative['rem_count'].value_counts())

print('Ben Count')
print('True Positive\n', true_positive['ben_count'].value_counts())
print('False Negative\n', false_negative['ben_count'].value_counts())
