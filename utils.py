import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)


# Resample the data according to fraud cases

def preprocessunEqualDistribution(data, sampling_ratio):
    X = data.ix[:, data.columns != 'Label']
    y = data.ix[:, data.columns == 'Label']

    number_records_fraud = len(data[data['Label'] == 1])
    fraud_indices = np.array(data[data['Label'] == 1].index)

    normal_indices = data[data['Label'] == 0].index
    random_normal_indices = np.random.choice(normal_indices,
                                             int((number_records_fraud * sampling_ratio) / (10 - sampling_ratio)),
                                             replace=False)

    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    under_sample_data = data.loc[under_sample_indices, :]

    X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Label']
    y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Label']

    # X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(
    # X_undersample,y_undersample,test_size=0.3, random_state=0,stratify = y_undersample)

    # fraud_train_indices = np.array(X_train_undersample[data['Label'] == 1].index)
    # genuine_train_indices = np.array(X_train_undersample[data['Label'] != 1].index)

    return X_undersample, y_undersample


def normalizeColumn(data, columns):
    for column in columns:
        data[column] = data[column] / data[column].max()
    return data


def plotConfusionMatrix(conf):
    sns.heatmap(conf, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
    plt.show()


def createBatches(data, numBatch):
    constData = data[data['Label'] == 1]
    dataToSplit = data[data['Label'] == 0]
    batchSize = int(dataToSplit.shape[0] / numBatch)
    dataToSplit = dataToSplit.ix[random.sample(list(dataToSplit.index), numBatch * batchSize)]
    batch_no_array = np.array([])

    for itr in range(1, numBatch + 1):
        batch_no_array = np.append(batch_no_array, [itr] * batchSize)

    np.random.shuffle(batch_no_array)
    dataToSplit.loc[:, 'batch'] = batch_no_array

    batches = []
    for itr in range(1, numBatch + 1):
        newBatch = pd.concat([dataToSplit[dataToSplit['batch'] == itr], constData], sort=False)
        newBatch.drop(['batch'], axis=1, inplace=True)
        batches.append(newBatch)

    return batches


def splitFraudNonFraud(data):
    nonFraud = data[data['Label'] == 0]
    fraud = data[data['Label'] == 1]

    return nonFraud, fraud


def filterDataByDate(data, start, end):
    if start is None and end is None:
        return None
    elif start is None:
        return data[(data['Tx date and time'] < end)]
    elif end is None:
        return data[(data['Tx date and time'] >= start)]
    else:
        return data[(data['Tx date and time'] >= start) & (data['Tx date and time'] < end)]


def fraudcount(data_, col):
    column = col
    labels = data_[column].astype('category').cat.categories.tolist()
    counts = data_[column].value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    df = pd.DataFrame({column: counts.index, 'fraud_count': counts.values})
    df['fraud_percentage'] = df['fraud_count'] / df['fraud_count'].sum()
    return counts, df


def plotpiechart(series):
    print(series)
    series.plot.pie()
    plt.show()
