import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2

data = pd.read_excel('data/TxReportForML_BetweenDates_New Report Jan 2017- Dec 2018.xlsx')


data_temp = data[['Tx count for remitter', 'Tx count for beneficiary',
                 'IS_First_Tx for Beneficiary', 'Return/Chargeback [0="No", 1="Yes"]']]

# print(data_temp.dtypes)

# data_temp = data_temp.drop(data_temp[(data_temp['Tx count for remitter'] > 20) &
#                                      (data_temp['Return/Chargeback [0="No", 1="Yes"]'] == 1)].index)


def chisq_of_df_cols(df, c1, c2):
    print('\n\nChisquare test for', c1, c2)
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return chi2_contingency(ctsum.fillna(0))


# print(chisq_of_df_cols(data_temp, 'Tx count for remitter', 'Return/Chargeback [0="No", 1="Yes"]'))

print(chi2(data_temp.drop('Return/Chargeback [0="No", 1="Yes"]', axis=1), data_temp['Return/Chargeback [0="No", 1="Yes"]']))
#
# print(chisq_of_df_cols(data_temp, 'Tx count for sending method', 'Return/Chargeback [0="No", 1="Yes"]'))
#
# print(chisq_of_df_cols(data_temp, 'Days count from the last tx', 'Return/Chargeback [0="No", 1="Yes"]'))
#
# print(chisq_of_df_cols(data_temp, 'IS_First_Tx for Beneficiary', 'Return/Chargeback [0="No", 1="Yes"]'))