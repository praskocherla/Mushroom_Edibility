

Original file is located at
    https://colab.research.google.com/drive/1gmwtE1DONbnB4zffDdc1ga-gDuJVX9Rq

The ([Dataset](https://archive.ics.uci.edu/dataset/73/mushroom)) focuses on various attributes of mushrooms, particularly those that are crucial for determining their edibility.Specifically, We aim to explore:
* Which attributes are most indicative of a mushroom's edibility?
* Are there any clear patterns or correlations among the features and the target variable (poisonous or edible)?
* Can a predictive model be developed to accurately identify edible and poisonous mushrooms based on their attributes?

By analyzing this dataset, I hope to gain insights into the characteristics that distinguish edible mushrooms from poisonous ones, potentially aiding in safer mushroom identification and consumption.

## Milestone 1 : EDA

Steps to be followed for the above milestone is :-
* Import Dataset and Library
* Data extraction from link
* Summary of Dataset (Descriptive statistics )
* Change datatypes
## Data Cleaning
* Detecting Null Values
* Null Value imputation
* Outlier Detection
* Outlier Removal
## Data Visualisation
* Univariate Analysis(Histograms and boxplots )
* Bivariate Analysis(barplots and scatterplots)
* One Hot encoding
* Correlation matrix
"""

##Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/drive')



# #importing the dataset
data=pd.read_csv('/content/drive/MyDrive/ALY6040/agaricus-lepiota.csv',names=["Target","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"])

dataset = data.copy()

data.head()

data.shape

data.info()

"""Replacing the coded values"""

data['Target']=data['Target'].replace(['e','p'],['edible','poisonous'])

data['Target'].value_counts()

## calculated the % of data for each target variables type
(data['Target'].value_counts())/len(data)*100

data['cap-shape']=data['cap-shape'].replace(['x','b','c','f','k','s'],['convex','bell','conical','flat','knobbed','sunken'])

## calculated the % of data for each cap-shape variables type
data['cap-shape'].value_counts()

(data['cap-shape'].value_counts())/len(data)*100

data['cap-surface']=data['cap-surface'].replace(['s','y','f','g'],['smooth','scaly','fibrous','grooves'])

data['cap-surface'].value_counts()

## calculated the % of data for each cap-surface variables type
(data['cap-surface'].value_counts())/len(data)*100

data['cap-color']=data['cap-color'].replace(['n','w','g','b','c','r','p','u','e','y'],['brown','white','gray','buff','cinnamon','green','pink','purple','red','yellow'])

data['cap-color'].value_counts()

## calculated the % of data for each cap-color variables type
(data['cap-color'].value_counts())/len(data)*100

data['bruises']=data['bruises'].replace(['t','f'],['bruises','no'])

data['bruises'].value_counts()

## calculated the % of data for each bruises variables type
(data['bruises'].value_counts())/len(data)*100

data['odor']=data['odor'].replace(['p','a','l','n','c','y','f','m','s'],
                                  ['pungent','almond','anise','none','creosote','fishy','foul','musty','spicy'])

data['odor'].value_counts()

## calculated the % of data for each odor variables type
(data['odor'].value_counts())/len(data)*100

data['gill-attachment']=data['gill-attachment'].replace(['a','d','f','n'],['attached','descending','free','notched'])

data['gill-attachment'].value_counts()

## calculated the % of data for each gill-attachment variables type
(data['gill-attachment'].value_counts())/len(data)*100

data['gill-spacing']=data['gill-spacing'].replace(['c','w','d'],['close','crowded','distant'])

data['gill-spacing'].value_counts()

## calculated the % of data for each gill-spacing variables type
(data['gill-spacing'].value_counts())/len(data)*100

data['gill-size']=data['gill-size'].replace(['n','b'],['narrow','broad'])

data['gill-size'].value_counts()

## calculated the % of data for each gill-size variables type
(data['gill-size'].value_counts())/len(data)*100

data['gill-color']=data['gill-color'].replace(['k','n','b','h','g','r','o','p','u','e','w','y'],
                                              ['black','brown','buff','chocolate','gray','green','orange','pink','purple','red', 'white','yellow'])

data['gill-color'].value_counts()

## calculated the % of data for each gill-color variables type
(data['gill-color'].value_counts())/len(data)*100

data['stalk-shape'] = data['stalk-shape'].replace(['e','t'],['enlarging','tapering'])

data['stalk-shape'].value_counts()

## calculated the % of data for each stalk-shape variables type
(data['stalk-shape'].value_counts())/len(data)*100

data['stalk-root'] = data['stalk-root'].replace(['b','c','u','e','z','r','?'],['bulbous','club','cup','equal', 'rhizomorphs','rooted',np.nan])

data['stalk-root'].value_counts()

## calculated the % of data for each stalk-root variables type
(data['stalk-root'].value_counts())/len(data)*100

data['stalk-surface-above-ring']=data['stalk-surface-above-ring'].replace(['f','y','k','s'],['fibrous','scaly','silky','smooth'])

data['stalk-surface-above-ring'].value_counts()

## calculated the % of data for each stalk-surface-above-ring variables type
(data['stalk-surface-above-ring'].value_counts())/len(data)*100

data['stalk-surface-below-ring']=data['stalk-surface-below-ring'].replace(['f','y','k','s'],['fibrous','scaly','silky','smooth'])

data['stalk-surface-below-ring'].value_counts()

## calculated the % of data for each stalk-surface-below-ring variables type
(data['stalk-surface-below-ring'].value_counts())/len(data)*100

data['stalk-color-above-ring']=data['stalk-color-above-ring'].replace(['n','b','c','g','o','p','e','w','y'],['brown','buff','cinnamon','gray','orange', 'pink','red','white','yellow'])

data['stalk-color-above-ring'].value_counts()

(data['stalk-color-above-ring'].value_counts())/len(data)*100

data['stalk-color-below-ring']=data['stalk-color-below-ring'].replace(['n','b','c','g','o','p','e','w','y'],['brown','buff','cinnamon','gray','orange', 'pink','red','white','yellow'])

data['stalk-color-below-ring'].value_counts()

(data['stalk-color-below-ring'].value_counts())/len(data)*100

data['veil-type']=data['veil-type'].replace(['p','u'],['partial','universal'])

data['veil-type'].value_counts()

(data['veil-type'].value_counts())/len(data)*100

data['veil-color']=data['veil-color'].replace(['n','o','w','y'],['brown','orange','white','yellow'])

data['veil-color'].value_counts()

(data['veil-color'].value_counts())/len(data)*100

data['ring-number']=data['ring-number'].replace(['n','o','t'],['none','one','two'])

data['ring-number'].value_counts()

(data['ring-number'].value_counts())/len(data)*100

data['ring-type']=data['ring-type'].replace(['c','e','f','l','n','p','s','z'],
 ['cobwebby','evanescent','flaring','large', 'none','pendant','sheathing','zone'])

data['ring-type'].value_counts()

(data['ring-type'].value_counts())/len(data)*100

data['spore-print-color']=data['spore-print-color'].replace(['k','n','b','h','r','o','u','w','y'],
 ['black','brown','buff','chocolate','green', 'orange','purple','white','yellow'])

data['spore-print-color'].value_counts()

(data['spore-print-color'].value_counts())/len(data)*100

data['population']=data['population'].replace(['a','c','n','s','v','y'],
 ['abundant','clustered','numerous', 'scattered','several','solitary'])

data['population'].value_counts()

(data['population'].value_counts())/len(data)*100

data['habitat']=data['habitat'].replace(['g','l','m','p','u','w','d'],
 ['grasses','leaves','meadows','paths', 'urban','waste','woods'])

data['habitat'].value_counts()

(data['habitat'].value_counts())/len(data)*100

data['ring-number']=data['ring-number'].replace(['none','one','two'],['0','1','2'])

data['ring-number'].value_counts()

(data['ring-number'].value_counts())/len(data)*100

"""Checking the missing values"""

data.isna().sum()

## % of missing value
data['stalk-root'].isna().sum()/len(data)*100

"""Converting categorical variables to numerical using one hot encoding"""



data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22]].head()

data_dummies=(pd.get_dummies(data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22]],columns=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22]].columns))

data_dummies=pd.concat([data_dummies,data['stalk-root']],axis=1)

data_dummies.head()

data_dummies['stalk-root'].isna().sum()/len(data)*100

data_dummies['stalk-root'] = data_dummies['stalk-root'].replace(['bulbous','club','cup','equal','rhizomorphs','rooted'],[1,2,3,4,5,6])

data_dummies['stalk-root'].value_counts()

data_dummies.head()

data_dummies.columns.tolist()

from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
imputed_data = knn_imputer.fit_transform(data_dummies)
imputed_df = pd.DataFrame(imputed_data,columns=data_dummies.columns)

imputed_df.head()

imputed_df['stalk-root'].value_counts()

import math
imputed_df['stalk-root']=imputed_df['stalk-root'].apply(math.ceil)

imputed_df['stalk-root'] = imputed_df['stalk-root'].replace([1,2,3,4,5,6],['bulbous','club','cup','equal','rhizomorphs','rooted'])

imputed_df['stalk-root'].value_counts()

imputed_df['stalk-root'].isna().sum()

data2=pd.concat([data.drop('stalk-root',axis=1),imputed_df['stalk-root']],axis=1)

data_dummies2=pd.get_dummies(imputed_df['stalk-root'])

imputed_df=pd.concat([imputed_df,data_dummies2],axis=1)

imputed_df.drop('stalk-root',axis=1,inplace=True)

imputed_df.drop('Target_edible',axis=1,inplace=True)

imputed_df.head()

"""EDA

Exploring the cap features for mushrooms
"""

fig, axes =plt.subplots(2,2,figsize=(20,10))
category_counts = np.array(data['cap-shape'].value_counts().index)
#print(category_counts)
sns.countplot(x='cap-shape',data=data,palette='bright',ax=axes[0,0],hue='Target',order=category_counts)
axes[0,0].set_title('Target ~ cap-shape')
category_counts = np.array(data['cap-surface'].value_counts().index)
sns.countplot(x='cap-surface',data=data,palette='Set3',ax=axes[0,1],hue='Target',order=category_counts)
axes[0,1].set_title('Target ~ cap-surface')
category_counts = np.array(data['cap-color'].value_counts().index)
sns.countplot(x='cap-color',data=data,palette='Set1',ax=axes[1,0],hue='Target',order=category_counts)
axes[1,0].set_title('Target ~ cap-color')
sns.countplot(x='bruises',data=data,palette='Set2',ax=axes[1,1],hue='Target')
axes[1,1].set_title('Target ~ bruises')

"""# Instruction
## Please write observations for above graphs

Exploring the gill features for mushrooms
"""

data.iloc[:,[6,7,8,9,10,11,12,13,14]].head()

fig, axes =plt.subplots(2,2,figsize=(20,10))
sns.countplot(x='gill-attachment',data=data,palette='bright',ax=axes[0,0],hue='Target')
axes[0,0].set_title('Target ~ gill-attachment')

sns.countplot(x='gill-spacing',data=data,palette='Set3',ax=axes[0,1],hue='Target')
axes[0,1].set_title('Target ~ gill-spacing')

sns.countplot(x='gill-size',data=data,palette='Set1',ax=axes[1,0],hue='Target')
axes[1,0].set_title('Target ~ gill-size')

category_counts = np.array(data['gill-color'].value_counts().index)
sns.countplot(x='gill-color',data=data,palette='Set2',ax=axes[1,1],hue='Target',order=category_counts)
axes[1,1].set_title('Target ~ gill-color')

"""# Instructions:-
## Please write observations for above graphs

# Instructions:-
* Create simlar graphs for other features like Stalk,veil,ring,...etc
* Note the Observations
"""

data.iloc[:,6:].head()

fig, axes =plt.subplots(2,2,figsize=(20,10))
sns.countplot(x='stalk-surface-above-ring',data=data,palette='bright',ax=axes[0,0],hue='Target')
axes[0,0].set_title('Target ~ stalk-surface-above-ring')

sns.countplot(x='odor',data=data,palette='Set3',ax=axes[0,1],hue='Target')
axes[0,1].set_title('Target ~ odor')

sns.countplot(x='stalk-surface-below-ring',data=data,palette='Set1',ax=axes[1,0],hue='Target')
axes[1,0].set_title('Target ~ stalk-surface-below-ring')

sns.countplot(x='stalk-surface-below-ring',data=data,palette='Set2',ax=axes[1,1],hue='Target')
axes[1,1].set_title('Target ~ stalk-surface-below-ring')

fig, axes =plt.subplots(1,2,figsize=(20,5),squeeze=False)

sns.countplot(x='veil-type',data=data, palette='bright',ax=axes[0,0], hue='Target')
axes[0,0].set_title('Target ~ veil-type')

sns.countplot(x='veil-color',data=data, palette='Set3',ax=axes[0,1],hue='Target')
axes[0,1].set_title('Target ~ veil-color')

fig, axes =plt.subplots(1,2,figsize=(20,5),squeeze=False)

sns.countplot(x='ring-number',data=data, palette='bright',ax=axes[0,0], hue='Target')
axes[0,0].set_title('Target ~ ring-number')

sns.countplot(x='ring-type',data=data, palette='Set3',ax=axes[0,1],hue='Target')
axes[0,1].set_title('Target ~ ring-type')

fig, axes =plt.subplots(1,2,figsize=(20,5),squeeze=False)

sns.countplot(x='spore-print-color', data=data, palette='bright', ax=axes[0,0], hue='Target')
axes[0,0].set_title('Target ~ spore-print-color')

sns.countplot(x='population',data=data, palette='Set3',ax=axes[0,1],hue='Target')
axes[0,1].set_title('Target ~ population')

"""# Correlation Plot"""

imputed_df.columns.tolist()

matrix=imputed_df.corr()
corr_matrix=pd.DataFrame(matrix[(matrix['Target_poisonous']>0.5) | (matrix['Target_poisonous']<-0.5)]['Target_poisonous'])
print(corr_matrix)

cols=corr_matrix.index.to_list()

plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=False, cmap='coolwarm', fmt='.2f', square=True)

"""### TRAIN TEST SPLIT"""

cols

X=imputed_df[cols[1:-1]]

Y=imputed_df[cols[0]]
X.info()

Y.info()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
X_train.shape

X_test.shape

Y_test.shape

Y_train.shape

"""### LOGISTIC REGRESSION

* Why Logistic Regression
* use of logistic regression
* advantages with respect to this project
"""

import statsmodels.api as sm
X = sm.add_constant(X)
model1=sm.Logit(Y_train,X_train).fit()

model1.summary()

model1.aic

"""Taking only significant variables"""

X=imputed_df[[cols[4],cols[8]]]

Y=imputed_df[cols[0]]
X.info()

Y.info()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
X_train.shape

X = sm.add_constant(X)
model1=sm.Logit(Y_train,X_train).fit()

model1.summary()

model1.aic

test_pred=model1.predict(X_test)
test_pred_class = (test_pred >= 0.7).astype(int)

test_pred_class[0:5]

from sklearn.metrics import accuracy_score,confusion_matrix,recall_score, precision_score,classification_report
accuracy1=accuracy_score(Y_test,test_pred_class)
print(accuracy1)

all1)recall1=recall_score(test_pred_class,Y_test)
print(rec

precision1=precision_score(Y_test,test_pred_class)
print(precision1)

con_matrix=confusion_matrix(Y_test,test_pred_class)
print(con_matrix)

classification_report(Y_test,test_pred_class)

"""### KNN model"""

X=imputed_df[cols[1:-1]]
Y=imputed_df[cols[0]]
X.info()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
X_train.shape

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
model2=knn.fit(X_train,Y_train)

test_pred2=model2.predict(X_test)

accuracy2=accuracy_score(Y_test,test_pred2)
print(accuracy2)

print(precision_score(Y_test,test_pred2))

print(recall_score(Y_test,test_pred2))

con_matrix2=confusion_matrix(Y_test,test_pred2)
print(con_matrix2)

classification_report(test_pred2,Y_test)

"""Create a desicion tree taking all the variables
* take entire dataset
* split again in test and train
* build a classification tree

### DECISION TREE CLASSIFIER
Considering the entire dataset
"""

data2.info()

from sklearn.model_selection import train_test_split
X= imputed_df.iloc[:, 1:]
Y = imputed_df.iloc[:, 0]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
X_train.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text,plot_tree
model4 = DecisionTreeClassifier(random_state = 56, max_depth = 2)
model4.fit(X_train,Y_train)

tree_rules = export_text(model4,feature_names=X_train.columns.to_list())
print(tree_rules)

Y_train.unique()

plt.figure(figsize=(12, 8))
plot_tree(model4, feature_names=X_train.columns.to_list(), class_names=Y_train.unique().astype('str'), filled=True)
plt.show()

test_pred4=model4.predict(X_test)

accuracy4=accuracy_score(Y_test,test_pred4)
print(accuracy4)

precision4=precision_score(Y_test,test_pred4)
print(precision4)

recall4=recall_score(Y_test,test_pred4)
print(recall4)

con_matrix=confusion_matrix(Y_test,test_pred4)
print(con_matrix)

classification_report(Y_test,test_pred4)

"""### SVM"""

from sklearn.svm import SVC
model3 = SVC(gamma='auto')
model3.fit(X_train,Y_train)

test_pred3=model3.predict(X_test)

accuracy3=accuracy_score(Y_test,test_pred3)
print(accuracy3)

precision3=precision_score(Y_test,test_pred3)
print(precision3)

recall3=recall_score(Y_test,test_pred3)
print(recall3)

con_matrix=confusion_matrix(Y_test,test_pred3)
print(con_matrix)

classification_report(Y_test,test_pred3)

"""###Random Forest"""

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Split the data into training and testing sets
target = imputed_df['Target_poisonous']
X_train,X_test,Y_train,Y_test=train_test_split(imputed_df.iloc[:, 1:],target,test_size=0.3)
X_train.shape

# Initialize RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model
rfc.fit(X_train, Y_train)

feature_importances = pd.DataFrame(rfc.feature_importances_,columns=['feature_importances'])
feature_importances.index = X_train.columns.to_list()
cummsum=feature_importances['feature_importances'].cumsum()

feature_importances.sort_values(by='feature_importances',ascending=False)

# Make predictions
Y_pred = rfc.predict(X_test)

# Print classification report
print(accuracy_score(Y_pred,Y_test))

precision3=precision_score(Y_pred,Y_test)
print(precision3)

print(recall_score(Y_pred,Y_test))

con_matrix=confusion_matrix(Y_pred,Y_test)
print(con_matrix)
