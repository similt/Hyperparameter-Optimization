import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars=pd.read_csv("imports-85.data",header=None,names=cols)
# the data file can be downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
print(cars.shape)
(cars.head())
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]
numeric_cars=numeric_cars.replace("?",np.nan)
numeric_cars.info()
numeric_cars=numeric_cars.astype('float')
numeric_cars.isnull().sum()
numeric_cars.dropna(subset=["price"],inplace=True)
numeric_cars.fillna(numeric_cars.mean(),inplace=True)
test=numeric_cars["price"]
numeric_cars=(numeric_cars-numeric_cars.min())/(numeric_cars.max()-numeric_cars.min())
numeric_cars["price"]=test
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def knn_train_test(train_col,test_col,df): 
    np.random.seed(1)
    df=df.reindex(np.random.permutation(df.index))
    kf = KFold(5, shuffle=True, random_state=1)
    knn = KNeighborsRegressor()
    mses=cross_val_score(knn, df[train_col], df[test_col],scoring="neg_mean_squared_error", cv=kf)
    rmse = np.sqrt(np.absolute(mses))      
    return rmse
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def knn_train_test(train_col,test_col,df): 
    np.random.seed(1)
    df=df.reindex(np.random.permutation(df.index))
    nhalf=int(len(df)/2)
    train_one=df[:nhalf]
    test_one =df[nhalf:]
    knn = KNeighborsRegressor()
    knn.fit(train_one[[train_col]], train_one[test_col])
    predictions=knn.predict(test_one[[train_col]])
    mse=mean_squared_error(test_one[[test_col]],predictions)  
    rmse= np.sqrt(mse)
    return rmse
col = numeric_cars.columns.drop('price')
dicta={}
for i in col: 
    result=knn_train_test(i,'price',numeric_cars)
    dicta[i]=result
    #print(i,result)
pd.Series(dicta).sort_values()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def knn_train_testk(train_col,test_col,df,k): 
    np.random.seed(1)
    df=df.reindex(np.random.permutation(df.index))
    nhalf=int(len(df)/2)
    train_one=df[:nhalf]
    test_one =df[nhalf:]
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_one[[train_col]], train_one[test_col])
    predictions=knn.predict(test_one[[train_col]])
    mse=mean_squared_error(test_one[[test_col]],predictions)  
    rmse= np.sqrt(mse)
    return rmse
col = numeric_cars.columns.drop('price')
dicta={}
kval=[1,3,5,7,9]
for i in col: 
    alist=[]
    for k in kval:
        result=knn_train_testk(i,'price',numeric_cars,k)
        alist.append(result)
    dicta[i]=alist
    #print(i,result)
pd.set_option('display.max_colwidth', None)
pd.Series(dicta)#.sort_values()

import matplotlib.pyplot as plt
%matplotlib inline
dict2={}
for i in dicta: 
    #print(dicta[i])
    
    x=[]
    y=[]
    k=1
    for alist in dicta[i]: 
        x.append(k)
        k=k+2 
        y.append(alist)
    plt.plot(x,y)
    dict2[i]=np.mean(y)
    
 
aa=pd.Series(dict2).sort_values(ascending=True)

def knn_train_test(train_col,test_col,df,k): 
    np.random.seed(1)
    df=df.reindex(np.random.permutation(df.index))
    nhalf=int(len(df)/2)
    train_one=df[:nhalf]
    test_one =df[nhalf:]
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_one[train_col], train_one[test_col])
    predictions=knn.predict(test_one[train_col])
    mse=mean_squared_error(test_one[test_col],predictions)  
    rmse= np.sqrt(mse)
    return rmse
col = numeric_cars.columns.drop('price')#.tolist()
#col =['normalized-losses','wheel-base']
#for i in col: 
    #result=knn_train_test(col,'price',numeric_cars)
prec=['price']
reco=aa.index.tolist()
col=[reco[0]]
print(type(col))
rmsdict={}
for i in range(1,6): 
    #print(reco[i])
    col.append(reco[i])
    if i>2: 
        continue
    else: 
        al=[]
        for k in range(1,26):
            result=knn_train_test(col,prec,numeric_cars,k)
            #print('Number of features',len(col),'k=',k,'RMSE = ',result)
            al.append(result)
        rmsdict[i]=al  
    print(col)
print(rmsdict)

for i in rmsdict: 
    #print(dicta[i])
    
    x=[]
    y=[]
    k=1
    for alist in rmsdict[i]: 
        x.append(k)
        k=k+2 
        y.append(alist)
    plt.plot(x,y)
    
  
