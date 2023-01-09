#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")


# In[2]:


# Drop columns with many NA and other rows with NA
data.drop("Age", axis=1, inplace=True)
data.drop("Cabin", axis=1, inplace=True)
data.dropna(inplace=True)


# In[3]:


# Split dataset in features and target variable and change categorical values to dummies
feature_cols = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]
categorical_cols = ["Pclass", "Sex", "Embarked"]
X = data[feature_cols] # Features
X = pd.get_dummies(X, columns = categorical_cols)
y = data.Survived # Target variable


# In[4]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train.head(10)


# In[5]:


# Create Decision Tree classifier object
dt = DecisionTreeClassifier( max_depth=3)

# Train Decision Tree classifier
dt = dt.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[6]:


from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = X.columns, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('titanic.png')
Image(graph.create_png())


# In[7]:


results = pd.read_csv("test.csv")
X_results = results[feature_cols]
X_results = pd.get_dummies(X_results, columns = categorical_cols)


# In[8]:


X_results.info()


# In[13]:


X_results[X_results["Fare"].isna()]


# In[14]:


X_results_cleaned = X_results.dropna()


# In[15]:


results_predict = dt.predict(X_results_cleaned)


# In[16]:


print(results_predict)


# In[19]:


passenger_id = results["PassengerId"]
print(list(passenger_id))


# In[21]:


final = pd.DataFrame({"PassengerId": passenger_id,
                      "Survived": list(results_predict[:152]) + [0] + list(results_predict[152:])})


# In[23]:


final.to_csv('submission.csv', index=False)