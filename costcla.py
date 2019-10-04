import pandas as pd
import utils
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from costcla.models import CostSensitiveRandomForestClassifier

data = pd.read_csv('data/Jan17_Dec18.csv')

data_test = data[(data['Tx date and time'] > '2018-10-31') & (data['Tx date and time'] <= '2018-12-31')]
data_train = data[(data['Tx date and time'] <= '2018-10-31') & (data['Tx date and time'] > '2016-12-31')]

data_train = data_train.drop(['Tx date and time'], axis=1)

X_train = data_train.drop(['Label', 'Tx date and time'], axis=1)
y_train = data_train['Label']

X_test = data_test.drop(['Label', 'Tx date and time'], axis=1)
y_test = data_test['Label']

clf = CostSensitiveRandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
plt.show()