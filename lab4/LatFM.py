import pandas as pd
import numpy as np


def make_validate(data):## Modify for cross-val
    val_idx = []
    val_obj = []
    rows, cols = data.shape
    counter = int(rows*cols/100*10)
    for i in range(0,counter):

        rand_idx1 = np.random.randint(0, rows)
        rand_idx2 = np.random.randint(0, cols)
        if(data[rand_idx1, rand_idx2] != 99):
            val_idx.append([rand_idx1, rand_idx2])
            val_obj.append(data[rand_idx1, rand_idx2])
            data[rand_idx1, rand_idx2] = 99
        else:
            counter = counter + 1
    return val_idx, val_obj, data##

def predict_rating(user_id,item_id):
    
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

def sgd(iterations = 10):
    last_mse = 99999
    for iteration in range(0,iterations):
        error = []
        for user_id in range(0,latent_user_preferences.shape[0]):
            for item_id in range(0,latent_item_features.shape[0]):
                rating = new_data[user_id][item_id]
                if(rating != 99 ):
                    err = train(user_id,item_id,rating)
                    error.append(err)
        mse = (np.array(error) ** 2).mean()
        if((last_mse - mse)<0.1):
            print (mse)
            break
        else:  
            print (mse)
            last_mse = mse
            
    return mse




              
df = pd.read_csv("jester-data-1.csv",index_col=False,header=None)
df = df.drop(df.columns[[0]], axis=1)
factor_err = np.zeros(5)

for i in range(1,10):##Decide later
    data = df.to_numpy(copy=True)
    print(df.head())
    
    n_features = i
    idx, values, new_data = make_validate(data)
    n_df = pd.DataFrame(new_data)
    print(n_df.head())
    latent_user_preferences = np.random.random((new_data.shape[0], n_features))
    latent_item_features = np.random.random((new_data.shape[1],n_features))
    sgd()
    predictions = latent_user_preferences.dot(latent_item_features.T)
    pred_err = np.zeros(len(values))
    
    for j in range(0,len(values)):
        r , c = idx[j]
        pred_err[j] = predictions[r][c] - values[j]

    factor_err[n_features -1] = (np.array(pred_err) ** 2).mean()
    print('Current Pred. Error:')
    print(factor_err[n_features -1])
    
print(factor_err)

    
    








    
