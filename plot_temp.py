import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

data = [1708, 1375, 939, 667, 446, 189, 135, 116, 62, 42 ]
bins = ['0.0 - 0.1', '0.1 - 0.2', '0.2 - 0.3', '0.3 - 0.4', '0.4 - 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8',
        '0.8 - 0.9', '0.9 - 1.0']

index = np.arange(len(data))

plt.bar(index, data)
plt.xticks(index, bins, fontsize=5, rotation=30, weight='bold')
plt.xlabel('Probability assigned to prediction')
plt.ylabel('No. of transactions')
plt.title('Probability Distribution Plot: Random Forest 50-50 Sampling')
plt.show()

