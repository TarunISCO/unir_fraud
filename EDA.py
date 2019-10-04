import pandas as pd
import matplotlib.pyplot as plt


def summary(df):
    print(df.describe())
    return


def class_distribution(df):
    count_classes = pd.value_counts(df['Class'], sort=True)
    print(count_classes)
    count_classes.plot(kind='pie', shadow=True, legend=True)
    plt.title("Transaction class distribution")
    plt.show()


def feature_distribution(df, features):
    for feature in features:
        value_counts = pd.value_counts(df[feature], sort=True)
        print(value_counts)
        value_counts.plot(kind='pie', shadow=True, legend=True)
        plt.title(feature, "distribution")
        plt.show()


def value_counts(df, features):
    vc = {}
    for feature in features:
        value_counts = pd.value_counts(df[feature], sort=True)
        print(value_counts)
        vc[feature] = value_counts

    return vc
