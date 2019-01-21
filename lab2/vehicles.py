import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Read the .csv file
df = pd.read_csv('./vehicles.csv')

#Scatter plot of vehicles.csv
splot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)
splot.axes[0,0].set_ylim(0,)
splot.axes[0,0].set_xlim(0,)
splot.savefig("scatterplot.png",bbox_inches='tight')
splot.savefig("scatterplot.pdf",bbox_inches='tight')

plt.clf()

#Histogram of the current fleet
cf_plot = sns.distplot(df.iloc[:,0],kde=False, rug=True).get_figure()
axes = plt.gca()
axes.set_xlabel('Current Fleet No') 
axes.set_ylabel('Current Fleet Count')

cf_plot.savefig("cf_histogram.png",bbox_inches='tight')
cf_plot.savefig("cf_histogram.pdf",bbox_inches='tight')

plt.clf()

#Histogram of the proposed fleet
pf_plot = sns.distplot(df.iloc[:,1].dropna(), kde=False, rug=True).get_figure()
axes = plt.gca()
axes.set_xlabel('Proposed Fleet') 
axes.set_ylabel('Proposed Fleet Count')

pf_plot.savefig("pf_histogram.png",bbox_inches='tight')
pf_plot.savefig("pf_histogram.pdf",bbox_inches='tight')

