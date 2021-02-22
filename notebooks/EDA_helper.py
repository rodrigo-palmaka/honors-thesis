import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from scipy import stats
import random
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


def format_df(figs, ratings, to_drop = []):
    """
    figs: Pandas df, features
    ratings: Pandas df, labels
    to_drop: list, cols to remove

    """

#     print(len(ratings["Ticker Symbol"].unique()))
    ratings1 = ratings[ratings['S&P Domestic Long Term Issuer Credit Rating'].notna()]
    ratings1 = ratings1[['Global Company Key','S&P Domestic Long Term Issuer Credit Rating', 'Data Date',
                        'Ticker Symbol']]

    print("Unique Companies in Ratings: ", len(ratings1["Ticker Symbol"].unique()))
    if len(to_drop) == 0:
        figs_1 = figs.drop(list(figs.columns[[2, 3,4,11,12, 14, 16, 18, 20, 21, 23, 26, 30, 33, 35, 36, 37, 38]]), axis=1).dropna()
    else:
        figs_1 = figs.drop(to_drop, axis=1).dropna()
    print("Unique Companies in Features: ", len(figs_1["Ticker Symbol"].unique()))

    lst3 = [value for value in ratings1["Ticker Symbol"].unique() if value in figs_1["Ticker Symbol"].unique()]
    print("Intersection of companies: ",len(lst3))

    companies = ratings1["Ticker Symbol"].unique()
    # companies
    quarts = ['1231', '0331', '0630', '0930']
    figs_1 = figs_1.loc[figs_1['Ticker Symbol'].isin(companies)]
    figs_1['Data Date'] = figs_1['Data Date'].astype(str)
    ratings1['Data Date'] = ratings1['Data Date'].astype(str)
    ratings1 = ratings1.loc[ratings1['Data Date'].str[-4:].isin(quarts)]
    return figs_1, ratings1


def join_numerical(figs_1, ratings1, numerical=[], ind_to_keep=[], years=[]):
    """
    in
    figs_1: Pandas df, features
    ratings1: Pandas df, labels
    numerical: List of str col. names of numerical columns to keep

    out
    x: np array
    y: np array
    merged: Pandas df
    """
    merged = pd.merge(figs_1, ratings1, how='inner', on=["Data Date", "Ticker Symbol"])
    merged = merged.rename(columns = {'S&P Domestic Long Term Issuer Credit Rating':'rating'})

    if len(numerical) == 0:
        numerical = ['Current Assets - Total', 'Cash','Long-Term Debt - Total',
        'Earnings Per Share (Basic) - Including Extraordinary Items - 12 Months Moving',
        'Invested Capital - Total - Quarterly','Inventories - Total', 'Pretax Income',
        'Operating Income Before Depreciation']
    vals = merged['Data Date'].str[:4]
    merged['year'] = vals
    # yrs =

    if len(years) != 0:
        merged = merged[merged['year'].isin(years)]
    # merged = merged.drop(['year'], axis=1)
    if len(ind_to_keep) == 0:
        ind_to_keep = [True]*len(merged)


    x = merged.loc[ind_to_keep, numerical].values
    y = merged.loc[ind_to_keep,['rating']].values
    full = numerical.copy()
    full.append('rating')
    full.append('year')
    full.append('Ticker Symbol')



    merged = merged.loc[ind_to_keep, full]


    return x, y, merged

def encode(Y, form='', custom={}):

    if custom:
        ## TODO write this
        pass

    else:
        if form == 'full':
            embed = {'BBB-':-1, 'BBB':0, 'A-':2, 'BBB+':1, 'AA+':7, 'AA':6, 'A':3, 'AA-':5, 'BB':-3, 'BB+':-2,
                                'AAA':8, 'B':-6, 'B+':-5, 'A+':4, 'BB-':-4, 'CCC+':-8, 'B-':-7}
            labels = ['CCC+','B-', 'B', 'B+', 'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+',
                   'AA-', 'AA', 'AA+', 'AAA']
        elif form == 'three':
            embed = {'BBB-':1, 'BBB':1, 'A-':2, 'BBB+':1, 'AA+':2, 'AA':2, 'A':2, 'AA-':2, 'BB':0, 'BB+':0,
                    'AAA':2, 'B':0, 'B+':0, 'A+':2, 'BB-':0, 'CCC+':0, 'B-':0}
            labels = ['low', 'med', 'high']
        elif form == "IG":
            embed = {'BBB-':1, 'BBB':1, 'A-':1, 'BBB+':1, 'AA+':1, 'AA':1, 'A':1, 'AA-':1, 'BB':0, 'BB+':0,
                    'AAA':1, 'B':0, 'B+':0, 'A+':1, 'BB-':0, 'CCC+':0, 'B-':0}
            labels = ['HY', 'IG']

        elif form == 'letters':
            embed = {'BBB-':3, 'BBB':3, 'A-':4, 'BBB+':3, 'AA+':5, 'AA':5, 'A':4, 'AA-':5, 'BB':2, 'BB+':2,
                    'AAA':6, 'B':1, 'B+':1, 'A+':4, 'BB-':2, 'CCC+':0, 'B-':1}
            labels = ['CCC', 'B', 'BB', 'BBB', 'A', 'AA','AAA']
        # elif form == 'letters ex CCC':
        else:
            embed = {'BBB-':3, 'BBB':3, 'A-':4, 'BBB+':3, 'AA+':5, 'AA':5, 'A':4, 'AA-':5, 'BB':2, 'BB+':2,
                    'AAA':6, 'B':1, 'B+':1, 'A+':4, 'BB-':2, 'CCC+':0, 'B-':1}
            labels = ['B', 'BB', 'BBB', 'A', 'AA','AAA']
            # colors = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', '#828282', '#17d8ff', '#1770ff']
        Y_emb = np.array([embed[i] for i in Y.T[0]]).ravel()

    return Y_emb, labels


def data_split(X, Y, full, split=[0.7, 0.1], method="numer", smote=False, seed=100, neigh=5):
    """
    split = [ratio of train, ratio of val]
    note: ratio of test = 1.0 - (ratio of train + ratio of val)
    """
    np.random.seed(seed)

    #### TODO: if smote, give a non-smote Validation set
    ## No Validation set
    if split[1] == 0.0:
        test_split = 1.0 - split[0]

        if method == 'numer':
            splits = int(np.round(split[0]*len(full)))
            cc = list(full.index).copy()
            np.random.shuffle(cc)
            train_inds = cc[:splits]
            test_inds = cc[splits:]
            X_train = X[train_inds, :]
            Y_train = Y[train_inds]
            X_test =  X[test_inds, :]
            Y_test =  Y[test_inds]

            X_val = []
            Y_val = []
        if smote:
            # ks = np.min([np.min(), 5])
            oversample = SMOTE(k_neighbors=neigh)
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)


    else:
        if method == 'numer':
            splits = int(np.round(split[0]*len(full)))
            splits1 = splits + int(np.round(split[1]*len(full)))
            cc = list(full.index).copy()
            np.random.shuffle(cc)
            train_inds = cc[:splits]
            val_inds = cc[splits:splits1]
            test_inds = cc[splits1:]

            X_train = X[train_inds, :]
            Y_train = Y[train_inds]
            X_val = X[val_inds, :]
            Y_val = Y[val_inds]

            X_test =  X[test_inds, :]
            Y_test =  Y[test_inds]

        if smote:
            # ks = np.min([np.min(), 5])
            oversample = SMOTE(k_neighbors=neigh)
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)



    return X_train, Y_train, X_val, Y_val, X_test, Y_test
