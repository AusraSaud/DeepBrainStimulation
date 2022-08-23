#Prediction of subthalamic deep brain stimulation motor outcomes for Parkinson‘s disease patients
#Radziunas A., Saudargiene A., et al. 2022

#Boxplots of selected radiomic features in two classes: 
#poor DBS outcome vs good/very good DBS outcome

import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
import os


datadir = "data"
datafile = os.path.join(datadir, "data_radiomics_features_selected_mRMR.csv")
df = pd.read_csv(datafile, index_col=0)

fig, axes = plt.subplots(5,4, figsize=(15, 20))

for i,el in enumerate(list(df.columns.values)[:-1]):
    a = df.boxplot(el, by='DBS motor outcome (1-poor, 2-good/very good)', ax=axes.flatten()[i])
    a.grid('on', which='major', linewidth=1)
    title = a.set_title("\n".join(wrap(el, 30)))
    title.set_y(1.05)

plt.tight_layout() 
plt.suptitle('')
figpath = os.path.join("figures", "Fig2_radiomic_features_boxplots.png")
fig.savefig(figpath, dpi=400, bbox_inches='tight', format='png')
