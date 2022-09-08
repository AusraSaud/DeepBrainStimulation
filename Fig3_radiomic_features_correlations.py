#Prediction of subthalamic deep brain stimulation motor outcomes for Parkinson‘s disease patients
#Saudargiene A., Radziunas A. et al. 2022

#Spearman correlation coefficients between the selected radiomic features


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues


plt.figure(figsize=(16, 6))
datadir = "data"
datafile = os.path.join(datadir, "data_radiomics_features_selected_mRMR.csv")
dataframe = pd.read_csv(datafile, index_col=0, usecols =[i for i in range(21)])

p_values = calculate_pvalues(dataframe)
p_values = np.where(p_values < 0.05, np.where(p_values < 0.01, '**', '*'), '')
np.fill_diagonal(p_values, '')
strings = p_values
results = dataframe.corr(method='spearman').to_numpy()

labels = (np.asarray(["{1:.2f}{0}".format(string, value)
                      for string, value in zip(strings.flatten(),
                                               results.flatten())])
         ).reshape(20, 20)


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(dataframe.corr(method='spearman'), vmin=-1, vmax=1, annot=labels, fmt='')

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right') 
figpath = os.path.join("figures", "Fig3_radiomic_features_correlations.png")
plt.savefig(figpath, dpi=300, bbox_inches='tight', format='png')


