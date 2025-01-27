import quandl 
import pandas as pd
import math ,datetime
import numpy as np
#import kaggle
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')

quandl.ApiConfig.api_key ='CYAnqL6MTy69KbdcNhzD'
df = quandl.get('WIKI/GOOGL')

#Viewing Option


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100 #high-low percentage change
df['PCT_CHANGE']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100 #percentage change
df=df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]

#Creation of Label
forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)
forecast_out=math.ceil(0.01*len(df))
df['label']=df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True) # shifting causes na in few rows
#print(df)

X=np.array(df.drop(['label'],axis=1))


X=X[:-forecast_out]
X_lately=X[-forecast_out:]

X=preprocessing.scale(X)
df.dropna(inplace=True)
Y=np.array(df['label'])
print(len(X),len(Y))
 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

clf=LinearRegression(n_jobs=-1)
clf.fit(X_train,Y_train)

with open('Linearegression.pickle','wb') as fo:
    pickle.dump(clf,fo)

pickle_in=open('Linearegression.pickle','rb')
clf=pickle.load(pickle_in)

accuracy_cv=clf.score(X_test,Y_test)
test_set=clf.predict(X_lately)

print(accuracy_cv)
print(test_set)

df['Forecast']=np.nan
