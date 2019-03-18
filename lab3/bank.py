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
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        t = "(%.2f)"%(cm[i, j])
        print (t)
        plt.text(j, i, t,
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
    x_train = df_dummies.iloc[:, 0:col-1]
    x_target = df_dummies.iloc[:, col-1:col]
    x_features = df_dummies.columns.values
    n_features = len(df_dummies.columns)
    #Get the CV score of the tree classifier.
    result = cross_val_score(cls, x_train, x_target , cv=skf,scoring = make_scorer(acc), n_jobs=-1)

    #Print the accuracy of the classifier.
    print("ACC: %0.2f (+/- %0.2f)" % (result.mean(), result.std()))

    #Fit the data
    cls.fit(x_train,x_target)

    #Confusion matrix
    prediction = cls.predict(x_train)
    cnf_matrix = confusion_matrix(x_target, prediction)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(len(set(x_target))), normalize = False,
                          title='Confusion matrix')

    plt.savefig("bankTreeConfusion.png",bbox_inches='tight')
    plt.savefig("bankTreeConfusion.pdf",bbox_inches='tight')

    #Get important features
    
    importances = cls.feature_importances_
    std = np.std([ExtraTreesClassifier.feature_importances_ for ExtraTreesClassifier in cls.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print(indices)
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(n_features):
        print("%d. %s (%f)" % (f + 1, x_features[indices[f-1]],  importances[indices[f-1]]))

    # Plot the feature importances of the forest
    fig = plt.figure()
    plt.title("Feature importances")  # just the top 10 features
    num_feat_to_plot = 10
    plt.bar(range(num_feat_to_plot), importances[indices[:num_feat_to_plot]],
           color="r", yerr=std[indices[:num_feat_to_plot]], align="center")
    plt.xticks(range(num_feat_to_plot), np.array(x_features)[indices[:num_feat_to_plot]])
    plt.xlim([-1, num_feat_to_plot])
    fig.set_size_inches(15,8)
    axes = plt.gca()
    axes.set_ylim([0,None])

    plt.savefig("bankImportances.png",bbox_inches='tight')
    plt.savefig("bankImportances.pdf",bbox_inches='tight')

#-------------------------------------------------------------------------------
# End of bank.py
#-------------------------------------------------------------------------------
