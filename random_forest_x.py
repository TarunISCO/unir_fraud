import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import utils

warnings.filterwarnings('ignore')

#######################################################################################
# Import Data
#######################################################################################
path = u'data/NewFeatures_.csv'
data = pd.read_excel('data/June/Transformed_ACH.xlsx')
print(data.shape)
data2 = pd.read_excel('data/June/Transformed_CC-DC.xlsx')
print(data2.shape)
data = pd.concat([data, data2]).reset_index(drop=True)
print(data.shape)


features = joblib.load('data/model_columns_new.pkl')
features.append('Label')
features.append('Tx date and time')
features.append('Amount')
features.append('Age')
features.append('Joining Time')
features.append('Sending Method Type')

# features.remove('Tx count for remitter_transformed')
# features.remove('Tx count for beneficiary_transformed')
# features.remove('Days count from the last tx_transformed')
features.remove('Age_transformed')
features.remove('Joining Time_transformed')

data = data[features]

# data.drop('Sending Method Type_IV', axis=1, inplace=True)

#######################################################################################
# Split training and testing data and preprocessing
#######################################################################################
data_test = data[(data['Tx date and time'] >= '2019-04-01') & (data['Tx date and time'] < '2019-06-01')]
data_train = data[(data['Tx date and time'] > '2017-12-31') & (data['Tx date and time'] < '2019-04-01')]


print('train and test data shape')
print(data_train.shape, data_test.shape)


data_train = data_train.drop(['Tx date and time', 'Amount', 'Sending Method Type'], axis=1)
# X_train = data_train.drop(['Label', 'Tx date and time'], axis=1)
# y_train = data_train['Label']


X_test = data_test.drop(['Label', 'Tx date and time', 'Amount', 'Sending Method Type'], axis=1)
y_test = data_test['Label']


print(X_test.columns)

########################################################################################
# Using Batches + Sampling
########################################################################################


clfs = []
sampling = 6
ind = 1

tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9, 10]}]

training_batches = utils.createBatches(data_train, 4)
for batch in training_batches:
    X, y = utils.preprocessunEqualDistribution(batch, sampling)

    rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
    rf_clf.fit(X, y)
    # print(rf_clf.best_estimator_.feature_importances_)
    # for col, f_imp in zip(X.columns, rf_clf.best_estimator_.feature_importances_):
    #     print(col, f_imp)

    clfs.append(rf_clf)
    filename = 'data/models/june2019/combined/RandomForest_' + str(sampling * 10) + '-' + str((10 - sampling) * 10) + '_june_' + str(
        ind)
    ind = ind + 1
    # joblib.dump(rf_clf, filename)

# clf1 = joblib.load('data/models/batches/model_final_60-40_1')
# clf2 = joblib.load('data/models/batches/model_final_60-40_2')
# clf3 = joblib.load('data/models/batches/model_final_60-40_3')
# clf4 = joblib.load('data/models/batches/model_final_60-40_4')
#
# clfsn = [clf1, clf2, clf3, clf4]


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
conf_img_name = 'observations/result/base_features_bs/' + str(sampling * 10) + '-' + str(
    (10 - sampling) * 10) + '_conf_matrix.png'
# plt.savefig(conf_img_name)
plt.show()

data_test['y_pred'] = y_pred_
data_test['y_prob'] = y_prob[:, 1]


data_ach = data_test[data_test['Sending Method Type'].isin(['BANK ACCOUNT'])]
data_cc = data_test[data_test['Sending Method Type'].isin(['CREDIT CARD', 'DEBIT CARD', 'ULINKCARD'])]

conf_c = confusion_matrix(data_cc['Label'], data_cc['y_pred'])
sns.heatmap(conf_c, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()

conf_a = confusion_matrix(data_ach['Label'], data_ach['y_pred'])
sns.heatmap(conf_a, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()

# data_test.to_csv('data/June/test_results.csv')

# label_df = pd.DataFrame(data=dict(y_test=y_test,
#                                   y_pred=y_pred_,
#                                   y_prob=y_prob[:, 1],
#                                   amount=data_test['Amount'],
#                                   sending_method=data_test['Sending Method Type']))
#
# bank_account_data = label_df[label_df['sending_method'].isin(['BANK ACCOUNT'])]
# credit_debit_data = label_df[label_df['sending_method'].isin(['DEBIT CARD', 'CREDIT CARD', 'ULINKCARD'])]


def plot_prediction_probablity(Y_pred_probability, title, file_name=None):
    fig = plt.figure()
    fig.set_size_inches(10, 8, forward=True)

    (n, bins, patches) = plt.hist(Y_pred_probability, bins=5, histtype='stepfilled')

    plt.title(title)
    plt.ylabel('No Of Transactions')
    plt.xlabel('Probability Assigned to Prediction ')
    plt.xticks(np.linspace(0, 1, 11))
    img_title = 'observations/distribution_new/50-50_' + title + '.png'
    # plt.savefig(img_title)
    plt.show()

    print(title)
    print(n, bins)
    # if (file_name != None):
    #     fig.savefig(os.path.join(file_name, title))
    # else:
    #     plt.show()


def displayStats(dist, lower, upper):
    print('\n\n\n')
    print('lower: ', lower)
    print('upper: ', upper)
    total = dist.shape[0]

    tp = dist[(dist['y_test'] == 1) & (dist['y_pred'] == 1)]
    fp = dist[(dist['y_test'] == 0) & (dist['y_pred'] == 1)]
    fn = dist[(dist['y_test'] == 1) & (dist['y_pred'] == 0)]
    tn = dist[(dist['y_test'] == 0) & (dist['y_pred'] == 0)]

    print('tp_shape', tp.shape)
    print('fn_shape', fn.shape)
    print('tn_shape', tn.shape)
    print('fp_shape', fp.shape)

    res = {'tp_mu': [tp[(tp['y_prob'] >= 0.5) & (tp['y_prob'] < upper)].shape[0],
                     (tp[(tp['y_prob'] >= 0.5) & (tp['y_prob'] < upper)].shape[0] / total) * 100],
           'tp_ut': [tp[(tp['y_prob'] >= upper) & (tp['y_prob'] <= 1.0)].shape[0],
                     (tp[(tp['y_prob'] >= upper) & (tp['y_prob'] <= 1.0)].shape[0] / total) * 100],
           'fp_mu': [fp[(fp['y_prob'] >= 0.5) & (fp['y_prob'] < upper)].shape[0],
                     (fp[(fp['y_prob'] >= 0.5) & (fp['y_prob'] < upper)].shape[0] / total) * 100],
           'fp_ut': [fp[(fp['y_prob'] >= upper) & (fp['y_prob'] <= 1.0)].shape[0],
                     (fp[(fp['y_prob'] >= upper) & (fp['y_prob'] <= 1.0)].shape[0] / total) * 100],
           'tn_bl': [tn[(tn['y_prob'] >= 0.0) & (tn['y_prob'] <= lower)].shape[0],
                     (tn[(tn['y_prob'] >= 0.0) & (tn['y_prob'] <= lower)].shape[0] / total) * 100],
           'tn_lm': [tn[(tn['y_prob'] > lower) & (tn['y_prob'] < 0.5)].shape[0],
                     (tn[(tn['y_prob'] > lower) & (tn['y_prob'] < 0.5)].shape[0] / total) * 100],
           'fn_bl': [fn[(fn['y_prob'] >= 0.0) & (fn['y_prob'] <= lower)].shape[0],
                     (fn[(fn['y_prob'] >= 0.0) & (fn['y_prob'] <= lower)].shape[0] / total) * 100],
           'fn_lm': [fn[(fn['y_prob'] > lower) & (fn['y_prob'] <= 0.5)].shape[0],
                     (fn[(fn['y_prob'] > lower) & (fn['y_prob'] <= 0.5)].shape[0] / total) * 100]}

    for k, v in res.items():
        print(k, v)

    print('\n\n\n')

    plot_prediction_probablity(Y_pred_probability=fp['y_prob'], title='false positive')
    plot_prediction_probablity(Y_pred_probability=fn['y_prob'], title='false negative')
    plot_prediction_probablity(Y_pred_probability=tp['y_prob'], title='true positive')
    plot_prediction_probablity(Y_pred_probability=tn['y_prob'], title='true negative')


def amount_description_cards(df, start_prob, end_prob):
    true_positive = df[(df['y_test'] == 1) & (df['y_pred'] == 1)]
    false_positive = df[(df['y_test'] == 0) & (df['y_pred'] == 1)]
    false_negative = df[(df['y_test'] == 1) & (df['y_pred'] == 0)]
    true_negative = df[(df['y_test'] == 0) & (df['y_pred'] == 0)]

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

    return fraud_saved_by_model, fraud_saved_by_third_party, genuine_dealt_by_model, genuine_dealt_by_third_party, fraud_amount_escaped, genuine_amount_troubled


def amount_statistics_account(df, cutoff):
    print('################################################################################################')
    print('\n\nAmount statistics for Bank account transactions with cutoff = ', cutoff)
    total = df['amount'].sum()
    print('Total amount in Bank transactions', total)

    tp = df[(df['y_test'] == 1) & (df['y_prob'] >= cutoff)]['amount']
    fp = df[(df['y_test'] == 0) & (df['y_prob'] >= cutoff)]['amount']
    fn = df[(df['y_test'] == 1) & (df['y_prob'] < cutoff)]['amount']
    tn = df[(df['y_test'] == 0) & (df['y_prob'] < cutoff)]['amount']

    print('\n\nFraud amount saved by the model\n', tp.describe())
    print('total amount, percentage = ', tp.sum(), tp.sum() * 100 / total)

    print('\n\nGenuine amount dealt by the model\n', tn.describe())
    print('total amount, percentage = ', tn.sum(), tn.sum() * 100 / total)

    print('\n\nFraud tx not detected\n', fn.describe())
    print('total amount, percentage = ', fn.sum(), fn.sum() * 100 / total)

    print('\n\nGenuine tx wrongly detected\n', fp.describe())
    print('total amount, percentage = ', fp.sum(), fp.sum() * 100 / total)


def amount_statistics_card(card_df, start_prob, end_prob):
    total_amount = card_df['amount'].sum()
    print('################################################################################################')
    print('\n\nAmount statistics for CC/DC transactions with probability cutoffs ', start_prob, end_prob)

    c_f_saved_model, c_f_saved_vendor, c_g_dealt_model, c_g_dealt_vendor, c_f_escaped, c_g_troubled = amount_description_cards(
        card_df, start_prob, end_prob)

    print(
        c_f_saved_model.sum() + c_f_saved_vendor.sum() + c_g_dealt_model.sum() + c_g_dealt_vendor.sum() + c_f_escaped.sum() + c_g_troubled.sum())

    print('total amount in CC/DC transactions', total_amount)

    print('\n\nFraud amount saved by the model\n', c_f_saved_model.describe())
    print('total amount, percentage = ', c_f_saved_model.sum(), c_f_saved_model.sum() * 100 / total_amount)

    print('\n\nFraud amount saved by the third party\n', c_f_saved_vendor.describe())
    print('total amount, percentage = ', c_f_saved_vendor.sum(),
          c_f_saved_vendor.sum() * 100 / total_amount)

    print('\n\nGenuine amount dealt by the model\n', c_g_dealt_model.describe())
    print('total amount, percentage = ', c_g_dealt_model.sum(),
          c_g_dealt_model.sum() * 100 / total_amount)

    print('\n\nGenuine amount dealt by the third party\n', c_g_dealt_vendor.describe())
    print('total amount, percentage = ', c_g_dealt_vendor.sum(),
          c_g_dealt_vendor.sum() * 100 / total_amount)

    print('\n\nFraud amount escaped\n', c_f_escaped.describe())
    print('total amount, percentage = ', c_f_escaped.sum(),
          c_f_escaped.sum() * 100 / total_amount)

    print('\n\nGenuine amount troubled\n', c_g_troubled.describe())
    print('total amount, percentage = ', c_g_troubled.sum(),
          c_g_troubled.sum() * 100 / total_amount)


# print('Sampling = ', str(sampling * 10) + '-' + str((10 - sampling) * 10))
#
# amount_statistics_card(credit_debit_data, 0.4, 0.6)
# amount_statistics_card(credit_debit_data, 0.3, 0.7)
# amount_statistics_card(credit_debit_data, 0.2, 0.8)
# amount_statistics_card(credit_debit_data, 0.1, 0.9)
# amount_statistics_card(credit_debit_data, 0.05, 0.95)
#
# print('\n\n\n\n\n')
#
# amount_statistics_account(bank_account_data, 0.5)
# amount_statistics_account(bank_account_data, 0.4)
# amount_statistics_account(bank_account_data, 0.3)
# amount_statistics_account(bank_account_data, 0.6)
# amount_statistics_account(bank_account_data, 0.7)
