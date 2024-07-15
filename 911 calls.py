#!/usr/bin/env python
# coding: utf-8

# In[808]:


#####################################################################
############### Part I - Importing
#####################################################################

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[809]:


df = pd.read_csv('911.csv')


# In[810]:


df.info()


# In[811]:


###############################################################
############### Part II - Duplicates
###############################################################


# In[812]:


df[df.duplicated()]                      #### no duplicates found, good for us


# In[813]:


##############################################################
################## Part III - Missing Data
##############################################################


# In[820]:


fig, ax = plt.subplots(figsize=(20,12))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### we have missing values inside 3 columns, seems like twp and address has very few missing values so in that case we can drop those few values
#### for zip we will have to find a way to make it work


# In[821]:


df.head()


# In[822]:


df.isna().any()


# In[823]:


df[df.twp.isnull()].count()                #### so we have 43 empty rows in twp, we will drop them because compared to our data which has almost 100,000 rows
                                           #### it wouldn't effect our data much


# In[824]:


df.dropna(subset=['twp'],inplace=True)


# In[825]:


df.isna().any()


# In[826]:


df[df.twp.isnull()].count()         #### took care of the null values inside twp


# In[827]:


fig, ax = plt.subplots(figsize=(20,12))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### now lets take care of addr missing values


# In[828]:


df[df.addr.isnull()]                #### i initially had intended to drop these null values but because its more then 500 rows its better to put a placeholder instead


# In[829]:


df.addr.fillna('Unknown',inplace=True)


# In[830]:


df[df.addr.isnull()]


# In[831]:


fig, ax = plt.subplots(figsize=(20,12))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


#### took care of addr column now we will move towards zip column


# In[832]:


df[df.zip.isna()]


# In[833]:


df.zip.isna().sum()                 #### number of null values in zip, can't drop obviously so we will use the placeholder here same way we did for addr feature column


# In[834]:


mean_zip = df.zip.mean()           #### because zip is numerical we can take advantage of mean and replace null values with them

mean_zip


# In[835]:


df.zip.fillna(mean_zip,inplace=True)


# In[836]:


df.info()                       #### nothing is empty now or missing


# In[837]:


fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


#### seems like our data is now clean


# In[663]:


####################################################################
######################### Part IV - Feature Engineering
####################################################################


# In[838]:


df.head()


# In[839]:


df.zip.head()         #### lets round zip column


# In[840]:


df.zip.round()


# In[841]:


df.zip = df.zip.round()


# In[842]:


df.zip.head()


# In[843]:


df.title.nunique()        #### number of unique titles


# In[844]:


df.title[0]               #### we will make a new column which extracts the reason from title column


# In[845]:


x = df.title[0]


# In[846]:


x.split(':')


# In[847]:


x.split(':')[0]


# In[848]:


df['Reasons'] = df.title.apply(lambda x:x.split(':')[0])             #### making a new column with reasons


# In[849]:


df.head()


# In[850]:


df.Reasons.unique()          #### power of lambda in python, now we have a clear reasons with only 3 categorical values


# In[851]:


df.Reasons.nunique()


# In[852]:


df.Reasons.value_counts()


# In[853]:


df['Num_Reasons'] = df.Reasons.map({'EMS':0,
                                    'Traffic':1,
                                    'Fire':2})


# In[854]:


custom = {'EMS':'purple',
          'Fire':'red',
          'Traffic':'green'}

sns.catplot(x='Reasons',data=df,kind='count',height=7,aspect=1.5,palette=custom)



#### majority are EMS related calls


# In[855]:


df.Num_Reasons.unique()


# In[856]:


df.zip.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('911 Reasons Graph')

plt.xlabel('Number of calls')

plt.ylabel('Zip numbers')

plt.ylim(df.zip.min(),df.zip.mean())


#### the amount of calls we getting from mean of zip is just astounding, somthing should be wrong in that zip code


# In[857]:


df.zip.max()


# In[858]:


df.zip.min()


# In[859]:


df[df.zip > 20000]            #### these are total outliers so we will just drop it as they are only 2 values


# In[860]:


df = df[df.zip <= 20000]


# In[861]:


df.zip.mean()                     #### mean of zip column


# In[862]:


df.zip.std()                      #### we have the std of + or - of 322 on either side of z score or std level


# In[863]:


custom = {'Traffic':'red',
          'EMS':'green',
          'Fire':'orange'}

g = sns.jointplot(x='lat',y='zip',data=df,hue='Reasons',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)

#### we see one area heavily crowded with traffic calls, may be we can do something with that information to reduce the emergency 911 calls


# In[864]:


custom = {'Traffic':'red',
          'EMS':'black',
          'Fire':'orange'}

g = sns.jointplot(x='lat',y='zip',data=df,hue='Reasons',palette=custom)

g.ax_joint.set_xlim(39.9, 40.5)

g.fig.set_size_inches(17,9)

#### there a huge peak in zip 19500 and lat 40.1-40.2


# In[865]:


custom = {'Traffic':'pink',
          'EMS':'grey',
          'Fire':'purple'}

sns.catplot(x='Num_Reasons',y='zip',data=df,kind='box',height=7,aspect=2,palette=custom,legend=True,hue='Reasons')


#### theres a pattern to EMS emergency and zip code, same with Fire and Traffic in relation to zip codes


# In[866]:


custom = {'Traffic':'pink',
          'EMS':'grey',
          'Fire':'purple'}

sns.catplot(x='Num_Reasons',y='lat',data=df,kind='box',height=7,aspect=2,palette=custom,legend=True,hue='Reasons')


#### same issue with lat here, something is going on with 40.1 lat and all these emergency calls


# In[867]:


#################################################################
################ Part V - Normal Distrubution
#################################################################


# In[868]:


mean_df = df.zip.mean()
std_df = df.zip.std()

print(mean_df,std_df)


# In[869]:


#### now we will make a very nice comprehensive standard deviation graph for zip column
from scipy.stats import norm


x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(12, 6))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')


#### this is very basic one but as we feeling fancy today so we will do a very comprehensive one


# In[870]:


#### Comprehensive time

x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(20, 7))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')

#### areas under the curve
plt.fill_between(x, y, where=(x >= mean_df - std_df) & (x <= mean_df + std_df), color='green', alpha=0.2, label='68%')
plt.fill_between(x, y, where=(x >= mean_df - 2*std_df) & (x <= mean_df + 2*std_df), color='orange', alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= mean_df - 3*std_df) & (x <= mean_df + 3*std_df), color='yellow', alpha=0.2, label='99.7%')

#### mean and standard deviations
plt.axvline(mean_df, color='black', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 3*std_df, color='yellow', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 3*std_df, color='yellow', linestyle='dashed', linewidth=1)

plt.text(mean_df, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, plt.gca().get_ylim()[1]*0.05, f'z=1    {mean_df + std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, plt.gca().get_ylim()[1]*0.05, f'z=-1   {mean_df - std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=2  {mean_df + 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-2 {mean_df - 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=3  {mean_df + 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-3 {mean_df - 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')


#### annotate the plot
plt.text(mean_df, max(y), 'Mean', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, max(y), '-1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, max(y), '+1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, max(y), '-2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, max(y), '+2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, max(y), '-3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, max(y), '+3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')

#### labels
plt.title('Zip distribution inside the 911 Dataset')
plt.xlabel('Zip')
plt.ylabel('Probability Density')

plt.legend()


#### its pretty clear from the normal distribution that zip swings + or - 259 each way, then we have z score + - till 3
#### from this we can look more into those zip codes and see how can we reduce the crime rate or prevent traffics


# In[871]:


#############################################################
######### Part VI - Timestamp
#############################################################


# In[872]:


#### this will be an eye opener because the things you can do with timestamp will blow your mind and here we will prove it
#### right now its pretty hard to see which days the crimes or emergencies are higher but after using timestamp its going to become very crystal

new_df = df.copy()

new_df['timeStamp'] = pd.to_datetime(new_df['timeStamp'])


# In[873]:


df.info()                        #### df is uneffected


# In[874]:


new_df.info()           #### now we have made the time column into date time so its going to be easier to manage


# In[875]:


x = new_df.timeStamp[0]           #### now because we have converted to timestamp, we have a lot of flexibilities
x


# In[876]:


x.hour


# In[877]:


x.minute                        #### we can call by individual attributes of timestamp, amazing


# In[878]:


new_df.timeStamp.head()


# In[879]:


x.month


# In[880]:


x.date()


# In[881]:


x.year


# In[882]:


x.dayofweek


# In[883]:


#### making a new column called hour based on the timestamp column
#### ignore the warning, as its recommended to use .loc but you can use either methods and I find this one much better 

new_df['hour'] = new_df['timeStamp'].apply(lambda x:x.hour)


# In[884]:


new_df['month'] = new_df.timeStamp.apply(lambda x:x.month)


# In[885]:


new_df['day_of_week'] = new_df.timeStamp.apply(lambda x:x.dayofweek)


# In[886]:


new_df['month_name'] = new_df.month.map({1:'Jan',
                         2:'Feb',
                         3:'Mar',
                         4:'Apr',
                         5:'May',
                         6:'Jun',
                         7:'Jul',
                         8:'Aug',
                         9:'Sep',
                         10:'Oct',
                         11:'Nov',
                         12:'Dec'})


# In[887]:


new_df.month.unique()


# In[888]:


new_df.month_name.unique()


# In[889]:


new_df['Day'] = new_df.day_of_week.map({0:'Mon',
                                     1:'Tue',
                                     2:'Wed',
                                     3:'Thr',
                                     4:'Fri',
                                     5:'Sat',
                                     6:'Sun'})


# In[890]:


new_df.head()                       #### we got our month, day of the week and hour columns from timestamp


# In[891]:


new_df.day_of_week.unique()


# In[892]:


new_df.Day.unique()


# In[893]:


custom = {'Traffic':'orange',
          'EMS':'red',
          'Fire':'purple'}


sns.catplot(x='Day',data=new_df,kind='count',hue='Reasons',height=7,aspect=2,palette=custom)


# In[894]:


new_df.hour.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('911 Calls Graph')

plt.xlabel('Number of calls')

plt.ylabel('Hour')


#### if we see closely then we see theres a dark black line running across hours 10-15, it means we get most of the calls in that hour


# In[895]:


custom = {'EMS':'red',
          'Fire':'purple',
          'Traffic':'green'}

sns.catplot(x='month_name',data=new_df,kind='count',hue='Reasons',height=7,aspect=2,palette=custom)


# In[896]:


custom = {'EMS':'red',
          'Fire':'purple',
          'Traffic':'green'}

sns.catplot(x='hour',data=new_df,kind='count',hue='Reasons',height=7,aspect=2,palette=custom)


#### this is something finally showing some kinda pattern, the crimes or emergency situations are low when its the morning time
#### and yes it makes sense because usually most of the people are sleeping


# In[897]:



pl = sns.FacetGrid(new_df,hue='Reasons',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'hour',fill=True)

pl.set(xlim=(0,new_df.hour.max()))

pl.add_legend()

#### hour 0-5 less calls, then 15-18 hour we see a peak in traffic related calls which makes sense because thats the time people are leaving their offices


# In[898]:


pl = sns.FacetGrid(new_df,hue='Day',aspect=4,height=4)

pl.map(sns.kdeplot,'hour',fill=True)

pl.set(xlim=(0,new_df.hour.max()))

pl.add_legend()


#### seems like the calls on weekends are higher even during hours 0-5 which makes sense its weekend


# In[899]:


pl = sns.FacetGrid(new_df,hue='Reasons',aspect=4,height=4)

pl.map(sns.kdeplot,'month',fill=True)

pl.set(xlim=(0,new_df.month.max()))

pl.add_legend()


#### pretty revealing that the number of calls increases for Traffic during Jan-Feb then regresses to the mean
#### also we dont have the data for month 9-11 so that's why we see everything going down to zero


# In[900]:


new_df.month.unique()                           #### no month 9,10,11


# In[901]:


new_df.drop(columns='e',inplace=True)


# In[902]:


new_df.head()


# In[903]:


#### lets make a quick basic corr heatmap to see the relatioship

corr = new_df.corr()


# In[904]:


corr.head()


# In[905]:


fig, ax = plt.subplots(figsize=(20,7)) 

sns.heatmap(corr,annot=True,linewidths=0.5,ax=ax,cmap='cividis')


#### honestly from this we can't deduce any strong correlation which makes sense but there is very weak positive correlation to Reasons and hour as well as lng


# In[906]:


new_df.groupby('month_name').count().plot(legend=True,figsize=(20,7),marker='o',markersize=14,markerfacecolor='black',linestyle='dashed',linewidth=2,color='red')


#### we can use the groupby to our advantage and see this very informative plot
#### seems like th calls are the highest during the first month of year Jan followed by summer month July


# In[907]:


new_df.month_name = new_df.month_name.astype('category')


# In[908]:


new_df.info()


# In[910]:




sns.lmplot(x='month',y='zip',data=new_df.groupby('month').count().reset_index(),height=7,aspect=2,line_kws={'color':'black'},scatter_kws={'color':'green'})


#### we see some linear relationship


# In[911]:


new_df['Date'] = new_df.timeStamp.apply(lambda x:x.date())


# In[912]:


new_df.head()


# In[913]:


new_df.groupby('Date').count().plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=10,linestyle='dashed',color='red')


#### same we see here as before, month Jan has the higest density of calls


# In[914]:


new_df.groupby('Date').count()['lat'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)


#### samething but we just selected lat from the groupby 


# In[915]:


new_df[new_df.Reasons == 'EMS'].groupby('Date').count()['lat'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)


#### we see that EMS related calls are at the peak during Jan-Feb months and is lowest in month 5


# In[916]:


new_df[new_df.Reasons == 'Traffic'].groupby('Date').count()['lat'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='black',markersize=10)


#### traffic calls are highest during new year month Jan and then declines substantially and is lowest during month 5


# In[917]:


new_df[new_df.Reasons == 'Fire'].groupby('Date').count()['lat'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='orange',color='black',markersize=10)


#### its the least calls compared to traffic and EMS but still peaks in month 1-3 and lowest in month 5


# In[918]:


new_df.groupby(by=['day_of_week','hour']).count()['Reasons'].unstack()          #### we use unstack to make into matrix form like here, we could have done with pivot table too but I love this approach

#### we did something very interesting here, we just are interested in Reasons so we grouped by Reasons and hour and day of the week and then unstack to form a matrix


# In[919]:


heat = new_df.groupby(by=['day_of_week','hour']).count()['Reasons'].unstack()


# In[920]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(heat,ax=ax,linewidths=0.5)


#### day of the week lets name it for better understanding


# In[921]:


new_df.groupby(by=['Day','hour']).count()['Reasons'].unstack()


# In[922]:


heat = new_df.groupby(by=['Day','hour']).count()['Reasons'].unstack()


# In[923]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### this is much much better representation, here we see that on weekends even during hours 0-5 theres some activities in 911 calls
#### then we have the higest during Friday and hour 16-17 followed by Tuesday and Thrusday of same hour which is suprising, seems like they are traffic related calls as we saw earlier


# In[924]:


new_df.groupby(by=['month_name','Day','hour']).count()['Reasons'].unstack().unstack()


# In[925]:


heat_2 = new_df.groupby(by=['month_name','Day','hour']).count()['Reasons'].unstack().unstack()


# In[931]:


fig, ax = plt.subplots(figsize=(30,15))

sns.heatmap(heat_2,ax=ax,linewidths=0.5)


# In[932]:


fig, ax = plt.subplots(figsize=(30,15))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')


#### I prefer viridis but a lot of people like the default one, so here we do both
#### this is extremely detailed informative map to see which day and which month did we receive the highest calls, I love it


# In[933]:



new_df.groupby(by=['Day','month_name']).count()['Reasons'].unstack()

#### lets just see month and day heatmap


# In[934]:


heat_3 = new_df.groupby(by=['Day','month_name']).count()['Reasons'].unstack()


# In[935]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(heat_3,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### seems like Jan and Sat is highest number of calls we get 


# In[938]:



sns.clustermap(heat_3,cmap='viridis')                 #### I try to stay away from cluster map as I prefer heat map and draw my own conclusion but we have it if we need it


# In[939]:


new_df.head()


# In[940]:


new_df.addr.nunique()


# In[941]:


#############################################
############ Model - Classification
############################################


# In[761]:


X = new_df.drop(columns=['desc','title','timeStamp','addr','Reasons','Num_Reasons','month_name','Day','Date'])


# In[762]:


X.head()


# In[763]:


y = new_df['Reasons']


# In[764]:


y.head()


# In[765]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[766]:


preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), ['twp']),
                                               ('num', StandardScaler(),['lat','lng','zip','hour','month','day_of_week'])
                                              ]
                                )


# In[767]:


from sklearn.pipeline import Pipeline


# In[768]:


from sklearn.linear_model import LogisticRegression


# In[769]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# In[770]:


from sklearn.model_selection import train_test_split


# In[771]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[772]:


model.fit(X_train,y_train)


# In[773]:


y_predict = model.predict(X_test)


# In[774]:


from sklearn import metrics


# In[775]:


metrics.accuracy_score(y_test,y_predict)             #### not bad for a very basic model


# In[776]:


print(metrics.classification_report(y_test,y_predict))


# In[777]:


from sklearn.ensemble import RandomForestClassifier           #### lets bring the randomforest


# In[778]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1,verbose=2))
])


# In[779]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train,y_train)')


# In[780]:


y_predict = model.predict(X_test)


# In[781]:


print(metrics.classification_report(y_test,y_predict))            #### slightly better


# In[782]:


metrics.confusion_matrix(y_test,y_predict)


# In[786]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['EMS','Fire','Traffic']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(20,11))

disp.plot(ax=ax)

#### not the best model because this dataset didn't have anything which had higher correlation,
#### I think the worst to predict is Fire which makes sense because those are in smaller numbers compared to others


# In[626]:


y_test.value_counts()


# In[627]:


y_train.value_counts()


# In[628]:


from sklearn.model_selection import GridSearchCV


# In[630]:


get_ipython().run_cell_magic('time', '', "\nparam_grid = {\n    'classifier__n_estimators': [100, 200, 300],\n    'classifier__max_depth': [None, 10, 20, 30],\n    'classifier__min_samples_split': [2, 5, 10],\n    'classifier__min_samples_leaf': [1, 2, 4]\n}\n\nmodel_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=3)\nmodel_grid.fit(X_train, y_train)")


# In[631]:


best_model = model_grid.best_estimator_


# In[632]:


best_model


# In[633]:


y_predict = best_model.predict(X_test)


# In[634]:


metrics.accuracy_score(y_test,y_predict)               #### went from 0.58 to 0.604 accuracy but now our recall for Fire has deteriorated


# In[635]:


print(metrics.classification_report(y_test,y_predict))


# In[636]:


from sklearn.neighbors import KNeighborsClassifier


# In[637]:


get_ipython().run_cell_magic('time', '', "\nk_range = range(1,100)\n\naccuracy = []\n\nfor i in k_range:\n    \n    model = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', KNeighborsClassifier(n_neighbors=i))\n    ]) \n    \n    model.fit(X_train,y_train)\n    \n    y_predict = model.predict(X_test)\n    \n    accuracy.append(metrics.accuracy_score(y_test,y_predict))")


# In[638]:


plt.figure(figsize=(15,7))

plt.plot(k_range,accuracy,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=10,markerfacecolor='black')

plt.xlabel('K Values')

plt.ylabel('Accuracy')


#### seems like the accuracy does go up when we increase the k value in KNN, the most ideal being k22 - k30
#### but still its no where close to the accuracy we got from gridsearch + randomforest


# In[ ]:


########################################################################################################################
############ We decided to conclude the modeling process as the improvement in accuracy has plateaued at 0.605.  #######
############ Although we considered exploring more advanced modeling techniques, our current computing resources #######
############ are not sufficient to efficiently handle 100,000 data points with these methods.  #########################
########### The GridSearchCV took over 12 hours to complete, highlighting the limitations of our computing power. ######
########### Therefore, we have chosen to halt further exploration for now.    ##########################################
########################################################################################################################

