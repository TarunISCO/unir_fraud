import pandas as pd
import numpy as np
import utils
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier


warnings.filterwarnings('ignore')

data = pd.read_csv('data/NewFeatures_.csv')

data_test = data[(data['Tx date and time'] > '2018-09-30') & (data['Tx date and time'] <= '2018-11-30')]
data_train = data[(data['Tx date and time'] > '2016-12-31') & (data['Tx date and time'] <= '2018-09-30')]

# print(data_train.shape, data_test.shape)


data_train = data_train.drop(['Tx date and time', 'Sending Method Type', 'Amount'], axis=1)
# For sampling
sampling = 9
X_train, y_train = utils.preprocessunEqualDistribution(data_train, sampling)

# X_train = data_train.drop(['Label'], axis=1)
# y_train = data_train['Label']

## Split without sampling
# X_train = data_train.drop(['Label'], axis=1)
# y_train = data_train['Label']

X_test = data_test.drop(['Label', 'Tx date and time', 'Amount', 'Sending Method Type'], axis=1)
y_test = data_test['Label']

clf1 = DecisionTreeClassifier()
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = KNeighborsClassifier()
# clf4 = SVC(C=10,kernel='rbf',max_iter=1000)
lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=DecisionTreeClassifier(),
                          use_probas=False, average_probas=False)

sclf.fit(X_train, y_train)

y_pred = sclf.predict(X_test)
y_prob = sclf.predict_proba(X_test)
y_pred_ = (y_prob[:, 1] > 0.5).astype(int)

conf = confusion_matrix(y_test, y_pred)

sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()

label_df = pd.DataFrame(data=dict(y_test=y_test,
                                  y_pred=y_pred_,
                                  y_prob=y_prob[:, 1],
                                  amount=data_test['Amount'],
                                  sending_method=data_test['Sending Method Type']))

bank_account_data = label_df[label_df['sending_method'].isin(['BANK ACCOUNT'])]
credit_debit_data = label_df[label_df['sending_method'].isin(['DEBIT CARD', 'CREDIT CARD', 'ULINKCARD'])]


def plot_prediction_probablity(Y_pred_probability, title, file_name=None):
    fig = plt.figure()
    fig.set_size_inches(10, 8, forward=True)

    (n, bins, patches) = plt.hist(Y_pred_probability, bins=5, histtype='stepfilled')

    plt.title(title)
    plt.ylabel('No Of Transactions')
    plt.xlabel('Probability Assigned to Prediction ')
    plt.xticks(np.linspace(0, 1, 11))
    plt.show()

    print(title)
    print(n, bins)


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


def amount_description_account(df):
    tp = df[(df['y_test'] == 1) & (df['y_pred'] == 1)]
    fp = df[(df['y_test'] == 0) & (df['y_pred'] == 1)]
    fn = df[(df['y_test'] == 1) & (df['y_pred'] == 0)]
    tn = df[(df['y_test'] == 0) & (df['y_pred'] == 0)]

    return tp['amount'], fp['amount'], fn['amount'], tn['amount']


def amount_statistics(card_df, account_df, start_prob, end_prob):
    print(data_test.shape, card_df.shape, account_df.shape)

    total_amount = data_test['Amount'].sum()
    c_f_saved_model, c_f_saved_vendor, c_g_dealt_model, c_g_dealt_vendor, c_f_escaped, c_g_troubled = amount_description_cards(
        card_df, start_prob, end_prob)
    a_f_saved_model, a_g_troubled, a_f_escaped, a_g_dealt_model = amount_description_account(account_df)

    print(
        c_f_saved_model.sum() + c_f_saved_vendor.sum() + c_g_dealt_model.sum() + c_g_dealt_vendor.sum() + c_f_escaped.sum() + c_g_troubled.sum())
    print(a_f_saved_model.sum() + a_g_troubled.sum() + a_f_escaped.sum() + a_g_dealt_model.sum())

    fraud_saved_model = c_f_saved_model.append(a_f_saved_model, ignore_index=True)
    genuine_dealt_model = c_g_dealt_model.append(a_g_dealt_model, ignore_index=True)
    fraud_escaped = c_f_escaped.append(a_f_escaped, ignore_index=True)
    genuine_troubled = c_g_troubled.append(a_g_troubled, ignore_index=True)
    fraud_saved_vendor = c_f_saved_vendor
    genuine_dealt_vendor = c_g_dealt_vendor

    print(
        fraud_saved_model.sum() + genuine_dealt_model.sum() + fraud_escaped.sum() + genuine_troubled.sum() + fraud_saved_vendor.sum() + genuine_dealt_vendor.sum())

    print('total amount', total_amount)

    print('\n\nFraud amount saved by the model\n', fraud_saved_model.describe())
    print('total amount, percentage = ', fraud_saved_model.sum(), fraud_saved_model.sum() * 100 / total_amount)

    print('\n\nFraud amount saved by the third party\n', fraud_saved_vendor.describe())
    print('total amount, percentage = ', fraud_saved_vendor.sum(),
          fraud_saved_vendor.sum() * 100 / total_amount)

    print('\n\nGenuine amount dealt by the model\n', genuine_dealt_model.describe())
    print('total amount, percentage = ', genuine_dealt_model.sum(),
          genuine_dealt_model.sum() * 100 / total_amount)

    print('\n\nGenuine amount dealt by the third party\n', genuine_dealt_vendor.describe())
    print('total amount, percentage = ', genuine_dealt_vendor.sum(),
          genuine_dealt_vendor.sum() * 100 / total_amount)

    print('\n\nFraud amount escaped\n', fraud_escaped.describe())
    print('total amount, percentage = ', fraud_escaped.sum(),
          fraud_escaped.sum() * 100 / total_amount)

    print('\n\nGenuine amount troubled\n', genuine_troubled.describe())
    print('total amount, percentage = ', genuine_troubled.sum(),
          genuine_troubled.sum() * 100 / total_amount)


amount_statistics(credit_debit_data, bank_account_data, 0.4, 0.6)
