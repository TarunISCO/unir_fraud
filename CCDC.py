import pandas as pd
from sklearn.externals import joblib
import utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

data = pd.read_excel('data/June/Transformed_CC-DC.xlsx')

print(data.shape)

features = joblib.load('data/model_columns_new.pkl')
features.append('Label')
features.append('Tx date and time')
features.append('Amount')
# features.remove('Tx count for remitter_transformed')
# features.remove('Tx count for beneficiary_transformed')
# features.remove('Days count from the last tx_transformed')

data = data[features]
data.drop('Sending Method Type_IV', axis=1, inplace=True)


data_test = data[(data['Tx date and time'] >= '2019-04-01')]
data_train = data[(data['Tx date and time'] >= '2018-01-01') & (data['Tx date and time'] < '2019-04-01')]
# data_train = data[data['Tx date and time'] < '2019-03-01']

print('train and test data shape')
print(data_train.shape, data_test.shape)

data_train = data_train.drop(['Tx date and time', 'Amount'], axis=1)

X_test = data_test.drop(['Label', 'Tx date and time', 'Amount'], axis=1)
y_test = data_test['Label']


clfs = []
sampling = 6
ind = 1

tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9, 10]}]

training_batches = utils.createBatches(data_train, 4)

for batch in training_batches:
    X, y = utils.preprocessunEqualDistribution(batch, sampling)

    rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
    rf_clf.fit(X, y)
    clfs.append(rf_clf)

    filename = 'data/models/june2019/CCDC/RandomForest_' + str(sampling * 10) + '-' + str((10 - sampling) * 10) + '_june_' + str(ind)
    ind = ind + 1
    joblib.dump(rf_clf, filename)


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

data_test['y_prob'] = y_prob[:, 1]

m_genuine = data_test[data_test['y_prob'] <= 0.05]
vendor = data_test[(data_test['y_prob'] > 0.05) & (data_test['y_prob'] < 0.95)]
m_fraud = data_test[data_test['y_prob'] >= 0.95]

print(f'Transactions below 0.05  = {m_genuine.shape[0]}, total amount =', m_genuine['Amount'].sum())
print(f'Transactions between 0.05 and 0.95 = {vendor.shape[0]}, total amount =', vendor['Amount'].sum())
print(f'Transactions above 0.95 = {m_fraud.shape[0]}, total amount =', m_fraud['Amount'].sum())

f_m_genuine = m_genuine[m_genuine['Label'] == 1]
g_m_genuine = m_genuine[m_genuine['Label'] == 0]

f_m_fraud = m_fraud[m_fraud['Label'] == 1]
g_m_fraud = m_fraud[m_fraud['Label'] == 0]

f_vendor = vendor[vendor['Label'] == 1]
g_vendor = vendor[vendor['Label'] == 0]


print('\nFrom 0.0 - 0.05')
print(f'Frauds = {f_m_genuine.shape[0]}, total amount =', f_m_genuine['Amount'].sum())
print(f'Genuine = {g_m_genuine.shape[0]}, total amount =', g_m_genuine['Amount'].sum())

print('\nFrom 0.95 - 1.0')
print(f'Frauds = {f_m_fraud.shape[0]}, total amount =', f_m_fraud['Amount'].sum())
print(f'Genuine = {g_m_fraud.shape[0]}, total amount =', g_m_fraud['Amount'].sum())


print('\nFrom 0.05 - 0.95')
print(f'Frauds = {f_vendor.shape[0]}, total amount =', f_vendor['Amount'].sum())
print(f'Genuine = {g_vendor.shape[0]}, total amount =', g_vendor['Amount'].sum())