import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, f_classif


file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'portland data science meetup', 'energy')
trainFile = os.path.join(file_dir, 'recs2015_public_v4.csv')


# C:\Users\cruze\Documents\portland data science meetup\energy

df = pd.read_csv(trainFile)

#%%

# drop imputation flag
df = df.loc[:, ~df.columns.str.startswith('Z')]

# # calculate the number of times to duplicate each sample
# weights_scaled = ((df['NWEIGHT']/df['NWEIGHT'].min())).astype(int)
# # duplicate the original indices based on weights_scaled
# resampled_idx = df.index.repeat(weights_scaled.values)
# # create dummy dataframe with duplicated index and join original data
# resampled_data = pd.DataFrame(index=resampled_idx, columns=['dummy']).join(df)
# # delete dummy column and reset index
# df = resampled_data.drop('dummy', axis=1).reset_index(drop=True)

#%%

## from John Burt's target feature engineering

# transform mapping
xform = {0:0, 1:3, 2:2, 3:1}

# remap the variables
df['SCALEBf'] = df['SCALEB'].map(xform)
df['SCALEGf'] = df['SCALEG'].map(xform)
df['SCALEEf'] = df['SCALEE'].map(xform)

def print_corr(df, var1, var2):
    """Calc Spearman corr and print result"""
    coef, pval = spearmanr(df[var1],df[var2])
    print('%s vs %s: coef=%1.3f, pval=%1.5f'%(var1,var2,coef,pval))

print('Spearman rank correlations:\n')
print_corr(df,'SCALEBf','SCALEGf')
print_corr(df,'SCALEBf','SCALEEf')
print_corr(df,'SCALEGf','SCALEEf')

print('\nHistograms:')

df[['SCALEBf','SCALEGf','SCALEEf']].hist()
plt.tight_layout()

# create PCA object
pca = PCA(n_components=1)

# do the PCA and save combined score to new 'hardship' column
df['hardship'] = pca.fit_transform(df[['SCALEBf','SCALEGf','SCALEEf']])

# scale the hardship values between 0 - 1
df['hardship'] = MinMaxScaler().fit_transform(
    df['hardship'].values.reshape(-1, 1))

# plot histo of new hardship score
df['hardship'].hist()
plt.title('"hardship" - PCA combined score of SCALEB, SCALEG, SCALEE')

#%%

# dropping SCALE and SCALEf columns
df = df.loc[:, ~df.columns.str.startswith('SCALE')]

# drop unique identifier
df = df.loc[:, df.columns != 'DOEID']

#%%

## input feature selection modified from John Burt

hardship_feat = ['COLDMA', 'HOTMA',
                 'NOACBROKE', 'NOACEL',
                 'NOHEATDAYSf', 'NOHEATDAYS', 
                 'NOHEATHELPf', 'NOHEATHELP',
                 'NOACDAYSf', 'NOACDAYS',
                 'NOACHELPf', 'NOACHELP',
                 'NWEIGHT', 'FREEAUDIT',
                 'NOHEATBROKE', 'NOHEATEL',
                 'PAYHELP', 'BENOTHER', 'NOHEATNG',
                 'ENERGYASST', 'ENERGYASST11', 
                 'ENERGYASST12', 'ENERGYASST15', 'ENERGYASST13',
                 'ENERGYASSTOTH', 'REBATEAPP', 'ENERGYASST14',
                 ]

btu_conversion = ['ELXBTU','NGXBTU','FOXBTU','LPXBTU']

# drop features related to hardship
# drop columns that start with Z
# drop columns that start with BRRW
dropcols = hardship_feat + ['hardship'] + btu_conversion
Xcols = []
for s in df.columns:
    if s not in dropcols:
        if s[0] != 'Z' and 'BRRW' not in s:
            Xcols.append(s)
            
X = df[Xcols]
print(X.shape)            

# drop features with string values
okidx = []
for i in range(X.shape[1]):
    if type(X.iloc[0,i]) != str:
        okidx.append(i)
X = X[X.columns[okidx]]

# drop columns with NaNs
X = X.dropna(axis=1) 

print(X.shape)

## our target variable
## make it binary
# threshold = 0.4
# y = df['hardship'] > threshold
# print('# samples: no hardship =', (~y).sum(), ', hardship =',y.sum())
# print('Percent: no hardship =', 100*(~y).sum()/len(y), 
#       ', hardship =',100*y.sum()/len(y))


# continuous target
y = df['hardship']

print(y.shape)


#%%


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


#%%

# select top k #
k = 5

# create and fit selector
selector = SelectKBest(f_classif, k)
selector.fit(X_train, y_train)
# get columns to keep
cols = selector.get_support(indices=True)
# create new dataframe with only desired columns, or overwrite existing
features_df_new = X_train.columns[cols]

print(features_df_new)


#%%

# sanity check
def plot_join_plot(df, feature, target):
    j = sns.jointplot(feature, target, data = df, kind = 'reg')
    j.annotate(stats.pearsonr)
    return plt.show()

train_df = pd.concat([X_train, y_train], axis=1)

plot_join_plot(train_df, 'KOWNRENT', 'hardship')



# %%
