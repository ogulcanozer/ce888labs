#_______________________________________________________________________________
# latFM.py | CE888 lab4     Ogulcan Ozer.  | UNFINISHED. check -> sss
#_______________________________________________________________________________
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def make_validate(data, shape):
    #Create validation set by changing the values of rated items.
    
    val_idx = []#index array
    val_obj = []#item array

    #Do for every cell in val. set
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):

            if(data[i][j] != 99):
                val_idx.append([i+shape,j])#Change relative pos. to original(added to the end of the training set)
                val_obj.append(data[i][j])
                data[i][j] = 99
    return val_idx, val_obj, data

def predict_rating(user_id,item_id):
    #predict the ratings from factors.
    user_preference = latent_user_preferences[user_id]
    item_preference = latent_item_features[item_id]
    return user_preference.dot(item_preference)

def train(user_id, item_id, rating, alpha = 0.001):

    #print item_id
    prediction_rating = predict_rating(user_id, item_id)
    err =  ( prediction_rating- rating );
    #print err
    user_pref_values = latent_user_preferences[user_id][:]
    latent_user_preferences[user_id] -= alpha * err * latent_item_features[item_id]
    latent_item_features[item_id] -= alpha * err * user_pref_values
    return err

def sgd(iterations = 5):
    last_mse = 99999
    for iteration in range(0,iterations):
        error = []
        for user_id in range(0,latent_user_preferences.shape[0]):
            for item_id in range(0,latent_item_features.shape[0]):
                rating = all_data[user_id][item_id]
                if(rating != 99 ):
                    err = train(user_id,item_id,rating)
                    error.append(err)
        mse = (np.array(error) ** 2).mean()
        if((last_mse - mse)<0.4):#Cut-off if the latest mse<0.4
            print (mse)
            break
        else:
            print (mse)
            last_mse = mse#Else save the latest and go on.
    return mse

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

#Import the data
df = pd.read_csv("jester-data-1.csv",index_col=False,header=None)
factor_err = np.zeros(3)


for i in range(1,3):##Decide later
    print(df.head())
    ## Number of features for the factorization
    n_features = 10
    ## Cross Validation splits
    K_fold = 10
    #**************************
    total_ratings = df.iloc[:,0].sum()
    print(df.iloc[:,0])
    df.sort_values([0], inplace=True)
    print(df.iloc[:,0])
    print(total_ratings)
    #***************************
    org_data = df.to_numpy(copy=True)
    y = org_data[:,0]
    X = org_data[:,1:]
    print(X.shape)
    print(y.shape)
    ## Stratified Cross Validation.
    sss = StratifiedShuffleSplit(n_splits=K_fold,random_state=0)
    sss.get_n_splits(X,y)
    cv_err = np.zeros(K_fold)
    cv_no = 0
    for train_index, test_index in sss.split(X, y):
        print('CROSVAL: ', cv_no)
        #Splitted validation set.
        val_data = pd.DataFrame(df.iloc[test_index]).to_numpy(copy=True)
        val_data = val_data[:,1:]
        v_df = pd.DataFrame(val_data)
        print('VAL: ', v_df.head())
        #Splitted training set.
        data = pd.DataFrame(df.iloc[train_index]).to_numpy(copy=True)
        data = data[:,1:]
        d_df = pd.DataFrame(data)
        print('DATA: ', d_df.head())
        #Save and change the validation set values to 99.
        idx, values, new_val = make_validate(val_data,data.shape[0])
        #idx = indices of the original values.
        #values = values
        nv_df = pd.DataFrame(new_val)
        print('NEW_VAL: ', nv_df.head())
        #Concatenate the changed validation set(new_val) to the training set.
        all_data = np.concatenate((data,new_val), axis=0)
        n_df = pd.DataFrame(all_data)
        print('ALL: ', n_df.head())
        ##
        latent_user_preferences = np.random.random((all_data.shape[0], n_features))
        latent_item_features = np.random.random((all_data.shape[1],n_features))
        sgd()
        #Get current cv predictions.
        predictions = latent_user_preferences.dot(latent_item_features.T)
        pred_err = np.zeros(len(values))
        #Calculate and save current CV score.
        for j in range(0,len(values)):
            r , c = idx[j]
            pred_err[j] = predictions[r][c] - values[j]

        cv_err[cv_no] = (np.array(pred_err) ** 2).mean()
        print('Current CV Error:')
        print(cv_err[cv_no])
        cv_no = cv_no + 1
    #Calculate and save average CV score for current number of features.
    factor_err[n_features -1] = cv_err.mean()
    print('Current Fact. Error:')
    print(factor_err[n_features -1])

print(factor_err)

#-------------------------------------------------------------------------------
# End of latFM.py
#-------------------------------------------------------------------------------
