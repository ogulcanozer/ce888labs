#_______________________________________________________________________________
# bank.py | CE888 lab3     Ogulcan Ozer. 
#_______________________________________________________________________________

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score as acc


#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv('./bank-additional-full.csv',sep=";")
    print(df.head())
    #Turn categorical values to numeric. ?
    df_dummies = pd.get_dummies(df)
    print(df_dummies.head())
    df_dummies.drop(['y_no', 'duration'], axis=1, inplace = True)
    print(df_dummies.head())
    #Get histograms of the values.
    d_plot = sns.distplot(df_dummies['y_yes'],kde=False, rug=True).get_figure()
    #Set plot labels
    axes = plt.gca()
    axes.set_xlabel('y_yes value') 
    axes.set_ylabel('y_yes Count')
    #Save the plots
    d_plot.savefig("y_yes_histogram.png",bbox_inches='tight')
    d_plot.savefig("y_yes_histogram.pdf",bbox_inches='tight')
    plt.show()

    #Initialize the tree classifier
    cls = ExtraTreesClassifier(n_estimators = 100,max_depth = 4)
    #Initialize the stratified Kfold CV.
    skf = StratifiedKFold(n_splits=10)
    
    col = len(list(df_dummies))
    #Get the CV score of the tree classifier.
    result = cross_val_score(cls, df_dummies.iloc[:, 0:col-1],df_dummies.iloc[:, col-1:col] , cv=skf,scoring = make_scorer(acc), n_jobs=-1)

    #Print the accuracy of the classifier.
    print("ACC: %0.2f (+/- %0.2f)" % (result.mean(), result.std()))
    ##look again##

#-------------------------------------------------------------------------------
# End of bank.py
#-------------------------------------------------------------------------------
