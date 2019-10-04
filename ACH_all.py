import itertools
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
Train models with different sets of features and sampling for multiple files
"""

for new_features, user_level, d in itertools.product([True, False], [True, False], ['jan2017', 'apr2017', 'jan2018']):
    data = pd.read_excel(f'data/july/{d}/Transformed-ACH.xlsx')

    print(data.shape)

    features = ['Age_transformed', 'Amount_transformed', 'IP_RISK_LEVEL_ID=4', 'FINAL_RESULT_EMAILAGE=FAIL',
                'FINAL_RESULT_EMAILAGE=SOFT FAIL', 'IP_RISK_LEVEL_ID=1', 'IP_RISK_LEVEL_ID=3', 'IP_RISK_LEVEL_ID=2',
                'Payer_IV', 'Beneficiary State_IV', 'IP_RISK_LEVEL_ID=6', 'Origin State_IV',
                'FINAL_RESULT_EMAILAGE=PASS', 'DOMAIN_NAME_IV', 'IP_RISK_LEVEL_ID=5', 'IP_USER_TYPE_IV',
                'Joining Time_transformed', 'Label', 'Tx date and time', 'Amount']

    if new_features:
        features.append('Tx count for remitter_transformed')
        features.append('Tx count for beneficiary_transformed')
        features.append('Days count from the last tx_transformed')

    if user_level:
        features.append('User Level_IV')

    data = data[features]

    data_test = data[(data['Tx date and time'] >= '2019-04-01')]
    data_train = data[(data['Tx date and time'] >= '2018-01-01') & (data['Tx date and time'] < '2019-04-01')]

    print('train and test data shape')
    print(data_train.shape, data_test.shape)

    data_train = data_train.drop(['Tx date and time', 'Amount'], axis=1)

    X_test = data_test.drop(['Label', 'Tx date and time', 'Amount'], axis=1)
    y_test = data_test['Label']

    tuned_parameters = [{'max_depth': [5, 6, 7, 8, 9, 10]}]
    training_batches = utils.createBatches(data_train, 4)

    for sampling in [5, 6, 7]:
        clfs = []

        for batch in training_batches:
            X, y = utils.preprocessunEqualDistribution(batch, sampling)

            rf_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='recall_macro')
            rf_clf.fit(X, y)
            clfs.append(rf_clf)


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

        ax = plt.axes()
        sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'}, ax=ax)

        plt_title = "ACH_basic{is_new}{is_user}_{samp}".format(is_new='+new' if new_features else '',
                                                               is_user='+user_level' if user_level else '',
                                                               samp='sampling=' + str(sampling))

        ax.set_title(plt_title)
        cnf_path = f'observations/july/{d}/ACH/{plt_title}.png'
        plt.savefig(cnf_path)
        plt.show()
