import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import warnings

import utils
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')
data = pd.read_csv('data/RecentData.csv')

########################################################################################
# Split training and testing data and preprocessing
########################################################################################
data_test = data[(data['Tx date and time'] > '2018-09-30') & (data['Tx date and time'] <= '2018-11-30')]
data_train = data[(data['Tx date and time'] <= '2018-09-30') & (data['Tx date and time'] > '2016-12-31')]

data_train = data_train.drop(['Tx date and time'], axis=1)
# X_train = data_train.drop(['Label', 'Tx date and time'], axis=1)
# y_train = data_train['Label']

X_train, y_train = utils.preprocessunEqualDistribution(data_train, 7)

X_test = data_test.drop(['Label', 'Tx date and time'], axis=1)
y_test = data_test['Label']

tuned_parameters = [{'n_estimators': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                     'max_depth': [1, 2, 3, 5, 10, 20, 50]}]
rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')

rf_clf.fit(X_train, y_train)
best_n = rf_clf.best_params_['n_estimators']
best_depth = rf_clf.best_params_['max_depth']
print(best_n, best_depth)

calibrated_classifier = CalibratedClassifierCV(RandomForestClassifier(n_estimators=best_n, max_depth=best_depth),
                                               method='isotonic', cv=5)
calibrated_classifier.fit(X_train, y_train)

y_pred = calibrated_classifier.predict(X_test)
y_prob = calibrated_classifier.predict_proba(X_test)

conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
conf_img_name = 'observations/distribution/50-50_conf_matrix.png'
plt.savefig(conf_img_name)
plt.show()

label_df = pd.DataFrame(data=dict(y_test=y_test, y_pred=y_pred, y_prob=y_prob[:, 1]))

true_positive = label_df[(label_df['y_test'] == 1) & (label_df['y_pred'] == 1)]
false_positive = label_df[(label_df['y_test'] == 0) & (label_df['y_pred'] == 1)]
false_negative = label_df[(label_df['y_test'] == 1) & (label_df['y_pred'] == 0)]
true_negative = label_df[(label_df['y_test'] == 0) & (label_df['y_pred'] == 0)]

print(true_positive.shape)
print(true_negative.shape)
print(false_positive.shape)
print(false_negative.shape)


def displayStats(tp, tn, fp, fn, lower, upper):
    print('\n\n\n')
    print('lower: ', lower)
    print('upper: ', upper)
    total = label_df.shape[0]

    # print('tp_shape', tp.shape)
    # print('fn_shape', fn.shape)
    # print('tn_shape', tn.shape)
    # print('fp_shape', fp.shape)

    fp_mu = [fp[(fp['y_prob'] < 0.5)]]
    print('fp < .5', fp_mu)

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


displayStats(true_positive, true_negative, false_positive, false_negative, 0.3, 0.7)
displayStats(true_positive, true_negative, false_positive, false_negative, 0.4, 0.6)
