import pickle

import pandas as pd
import utils
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier


data = pd.read_csv('data/NewFeatures.csv')

data_test = data[(data['Tx date and time'] > '2018-09-30') & (data['Tx date and time'] <= '2018-11-30')]
data_train = data[(data['Tx date and time'] > '2016-12-31') & (data['Tx date and time'] <= '2018-09-30')]


data_train = data_train.drop(['Tx date and time'], axis=1)
# For sampling
X_train_stacking, y_train_stacking = utils.preprocessunEqualDistribution(data_train, 5)

X_train_rf = data_train.drop('Label', axis=1)
y_train_rf = data_train['Label']

## Split without sampling
# X_train = data_train.drop(['Label'], axis=1)
# y_train = data_train['Label']

X_test_stacking = data_test.drop(['Label', 'Tx date and time'], axis=1)
y_test_stacking = data_test['Label']

clf1 = DecisionTreeClassifier()
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = KNeighborsClassifier()
# clf4 = SVC(C=10,kernel='rbf',max_iter=1000)
lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=DecisionTreeClassifier(),
                          use_probas=False, average_probas=False)

sclf.fit(X_train_stacking, y_train_stacking)

y_pred_stacking = sclf.predict(X_test_stacking)

conf = confusion_matrix(y_test_stacking, y_pred_stacking)

sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()

temp = X_test_stacking
temp['Label'] = y_pred_stacking.tolist()

test_rf = temp[temp['Label'] == 1]

X_test_rf = test_rf.drop('Label', axis=1)
y_test_rf = test_rf['Label']

# clf = RandomForestClassifier(random_state=1)
clf = pickle.load(open('data/models/NewDataRF_Sam_CodesIVSIL_100.pkl', 'rb'))
clf.fit(X_train_rf, y_train_rf)


y_pred_rf = clf.predict(X_test_rf)

conf = confusion_matrix(y_test_rf, y_pred_rf)

sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()
