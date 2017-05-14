
# coding: utf-8

# Assignment 1 - Machine Learning for Data Analysis
# ======

# Description of Dataset
# ----

# ## Source ##
# For this assignment the Statlog (Heart) Data Set ([Link](https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29)) obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html) is used. 
# 

# ## Description
# ### Attribute Information:
#  1. age
#  2. sex
#  3. chest pain type  (4 values)
#  4. resting blood pressure
#  5. serum cholestoral in mg/dl
#  6. fasting blood sugar > 120 mg/dl
#  7. resting electrocardiographic results  (values 0,1,2)
#  8. maximum heart rate achieved
#  9. exercise induced angina
#  10. oldpeak = ST depression induced by exercise relative to rest
#  11. the slope of the peak exercise ST segment
#  12. number of major vessels (0-3) colored by flourosopy
#  13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
#  
# ### Variable to be predicted
# Absence (1) or presence (2) of heart disease
# 
# # Code

# In[15]:

#Importing Pandas
from pandas import Series, DataFrame
import pandas as pd

#Importing NumPy
import numpy as np

# Matplotlib to plot
import matplotlib.pylab as plt

#Importing sklearn 
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics


# Reading the dataset as raw_data as assigning names of attributes to each coloumn. Also visualising at top five rows of the data. 

# In[16]:

raw_data = pd.read_csv("heart.dat", delimiter=' ', names = ["age", "sex", "chestPainType", "RestBP", "SerumCholesterol", "FastingBP", "RestingECG", "MaxHR", "ExerciseInduceAgina", "Oldepeak", "SlopSTSegment", "NoVessels", "Thal", "Result"])

raw_data.head()


# In[17]:

raw_data.describe()


# Creating dataframe without the target (col name = results)

# In[18]:

predictors = raw_data.drop(["Result"], axis=1)
#Shape of the data.
predictors.shape


# Storing target values in new array

# In[19]:

targets = raw_data.Result


# Splitting data into training and test set 60:40. 

# In[20]:

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)


# In[21]:

print (pred_train.shape,pred_test.shape,tar_train.shape,tar_test.shape)


# Defining and fitting the decision tree

# In[22]:

clf = DecisionTreeClassifier()
clf = clf.fit(pred_train, tar_train)


# making preditions for the test set. 

# In[23]:

predictions=clf.predict(pred_test)


# Printing the confusion metrics. 

# In[24]:

sklearn.metrics.confusion_matrix(tar_test,predictions)


# **The results shows the model has correctly predicted 48 True Postive and 32 True Negatives. But, there are 28 False predictions (14 each in FP and FN). **

# In[13]:

sklearn.metrics.accuracy_score(tar_test, predictions)


# ** Overall accurace of the model is around 79%. **

# ## Plotting Decision Tree.

# In[25]:

from sklearn import tree
from io import StringIO
from IPython.display import Image

out = StringIO()
tree.export_graphviz(clf, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())


# # Explaination
# 
# It is clear from the above tree the Decision Tree classfication is able to classfy the data with around 80% accuracy. The Most important attribute is Chest Pain Type. 

# In[ ]:



