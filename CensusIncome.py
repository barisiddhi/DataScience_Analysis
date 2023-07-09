import  numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error,roc_auc_score,r2_score


Census_data=pd.read_csv("C:/Users/Siddhi Bari/Downloads/adult.csv")
print(Census_data.shape)

#First Data
print(Census_data.head())
print(Census_data.info())

#Data Analysis and Visualization
print(Census_data.describe())

#check for the null values
print(Census_data.isnull().sum())

Census_data['income'].value_counts()
Census_data['sex'].value_counts()
Census_data['native.country'].value_counts()
Census_data['workclass'].value_counts()
Census_data['occupation'].value_counts()
#Census_data['martial.status'].value_counts()

#corrmat = Census_data.corr()
#print(corrmat)

#Visualize the distribution of income
sns.countplot(x='income', data=Census_data)
plt.title('Distribution of income')
plt.xlabel('income')
plt.ylabel('count')
plt.show()

sns.histplot(Census_data['age'],bins=20)
plt.title('Distribution of Age')
plt.xlabel('age')
plt.ylabel('count')
plt.show()

#Dropping the values
#drop Education
Census_data=Census_data.drop(['education','fnlwgt'],axis=1)
Census_data.head(1)

#replacing
#Census_data.replace('?', np.NaN.inplace = True)
#Census_data.head()


le = LabelEncoder()
Census_data['workclass']=le.fit_transform(Census_data['workclass'])
Census_data['occupation']=le.fit_transform(Census_data['occupation'])
Census_data['relationship']=le.fit_transform(Census_data['relationship'])
Census_data['race']=le.fit_transform(Census_data['race'])
Census_data['sex']=le.fit_transform(Census_data['sex'])
Census_data['native.country']=le.fit_transform(Census_data['native.country'])
Census_data['income']=le.fit_transform(Census_data['income'])
Census_data.head()

#barplot of income and Age
ploat= plt.figure(figsize=(5,5))
sns.barplot(x='income' ,y='age',data=Census_data)
plt.show()

ploat= plt.figure(figsize=(5,5))
sns.barplot(x='income' ,y='race',data=Census_data)
plt.show()



#training and testing sets
x = Census_data.drop(['income'],axis=1)
y =Census_data['income']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)


#Training the model
x_train, x_test ,y_train, y_test = train_test_split(x,y, test_size=0.2)
print(y.shape,y_train.shape,y_test.shape)

#gd = GaussianNB
#gd.fit(x_train,y_train)
#GaussianNB.fit()
#y_preb = gb.predict(y_test)'
'''plt.scatter(train_x, train_y)
plt.show()
plt.scatter(test_x, test_y)
plt.show()'''


y=Census_data.age
x=Census_data.drop('age',axis=1)

from sklearn.linear_model import LinearRegression as lm
#model=lm.fit(x_train,y_train)
plt.xlabel('True values')
plt.ylabel('Predictions')


