#_______________________________________________________________________________
# power.py | CE888 lab2     Ogulcan Ozer. 
#_______________________________________________________________________________

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------

def power(sample1, sample2, reps, size, alpha):
    counter = 0
    for i in range(0, reps):
        
        new_sample1 = np.random.choice(sample1, size=len(sample1), replace=True)
        new_sample2 = np.random.choice(sample2, size=len(sample2), replace=True)

        m1 = np.mean(new_sample1)
        m2 = np.mean(new_sample2)

        p = perm_test(new_sample1, new_sample2, size)
        
        if(p > (1 - alpha)):
            counter = counter + 1
    return counter/reps
    
def perm_test(sp1, sp2, size):
    s1 = sp1
    s2 = sp2
    counter = 0

    for i in range(0,20000):
        combined = s1 + s2
        perm_combined = np.random.choice(combined, size=len(combined), replace=True)
        tmp = array_split(perm_combined, 2, axis=0)
        s1 = tmp[0]
        s2 = tmp[1]
        tperm = np.mean(s1) - np.mean(s2)
        if( tperm > size ):
            counter = counter + 1

    return counter/20000
    
#-------------------------------------------------------------------------------
# End of power.py
#-------------------------------------------------------------------------------
