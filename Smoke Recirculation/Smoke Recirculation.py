#!/usr/bin/env python
# coding: utf-8

# This study uses a set of simulation results obtained using parametric simulations in Fire Dynamics Simulator (FDS), which is a Computational Fluid Dynamics (CFD) simulator, to study smoke reciruclation from exhaust shafts into supply shafts for underground train stations.
# 
# A full CFD simulation usually has a calculation time of a few days, and may take up to a week or two, which is resource intensive and may require multiple iterations.
# 
# This aim of study is to attempt to use build a model using logistic regression to make predictions on a preliminary design which does not have recirculation without performing a full CFD simulation, so as to allow the project to progress quicker and save time and cost.
# 
# Note: The data used in this study is not available online and is obtained from my current workplace.

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('smoke recirculation_cleaned.csv')


# In[3]:


df.head()


# The dataset is will be cleaned up by removing unused parameters.
# Note: parameters removed are those which are deemed insignificant (as determined by engineering analysis prior to this study) or used for sorting.

# In[4]:


df.drop('St', axis=1, inplace=True)
df.drop('Spy', axis=1, inplace=True)
df.drop('Spx', axis=1, inplace=True)
df.drop('Sny', axis=1, inplace=True)
df.drop('Snx', axis=1, inplace=True)
df.drop('Et', axis=1, inplace=True)
df.drop('Epy', axis=1, inplace=True)
df.drop('Epx', axis=1, inplace=True)
df.drop('Eny', axis=1, inplace=True)
df.drop('Enx', axis=1, inplace=True)
df.drop('Supply Vent Facing Direction', axis=1, inplace=True)
df.drop('Exhaust Vent Facing Direction', axis=1, inplace=True)
df.drop('Wind Angle', axis=1, inplace=True)
df.drop('EC', axis=1, inplace=True)


# A check is carried out to ensure there are no missing values and type of data is correct.

# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# Create variable 'Distance Between Vents' which uses the x,y,z displacement to return the distance.
# The variables containing the x,y,z displacements are then dropped.

# In[7]:


df['Distance Between Vents']=np.emath.sqrt(pow(df['x Distance Between Vents'],2)+pow(df['y Distance Between Vents'],2)+pow(df['Vertical Distance Between Vents'],2))
df.drop('x Distance Between Vents', axis=1, inplace=True)
df.drop('y Distance Between Vents', axis=1, inplace=True)
df.drop('Vertical Distance Between Vents', axis=1, inplace=True)


# In[8]:


df.head()


# In[9]:


df.describe()


# The presence of smoke causes Vis (visibility) to drop.
# As part of an engineering analysis, when Vis falls below 30, it can be said that there is smoke recirculation.

# In[10]:


sns.lmplot(x='Distance Between Vents',y='Recirculation',data=df)


# From the graph above, we can infer that the larger the distance between the supply and exhaust vents, the less likely recirculation occured.

# In[11]:


df.corr()


# Moving on to model building

# In[12]:


from sklearn.model_selection import train_test_split # splitting the data
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import jaccard_score as jss # evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import log_loss # evaluation metric


# Define 'x' as the independant variable and 'y' as the dependant variable.

# In[13]:


x = np.asarray(df[['Supply Vent Flow Speed','Exhaust Vent Flow Speed','Wind Speed','Fire Size','Distance Between Vents']])
y = np.asarray(df['Recirculation'])
print('x sample :', x[:5])
print('y sample :', y[:10])


# In[14]:


x = StandardScaler().fit(x).transform(x)
print('x sample :', x)


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)
print('x_train samples :', x_train[:5])
print('x_test samples :', x_test[:5])
print('y_train samples :', y_train[:10])
print('y_test samples :', y_test[:10])


# In[16]:


lr = LogisticRegression(C = 0.1, solver = 'liblinear')
lr.fit(x_train,y_train)
print(lr)


# In[17]:


yhat = lr.predict(x_test)
yhat_prob = lr.predict_proba(x_test)
print('yhat samples :', yhat[:10])
print('yhat_prob samples:', yhat_prob[:10])


# Model evaluation

# In[18]:


print('Jaccard Similarity Score of our model is {}'.format(jss(y_test,yhat).round(2)))


# In[19]:


print('Precision Score of our model is {}'.format(precision_score(y_test, yhat).round(2)))


# In[20]:


print('Log Loss of our model is {}'.format(log_loss(y_test, yhat).round(2)))


# In[21]:


print(classification_report(y_test, yhat))


# Attempt to plot the confusion matrix without the use of seaborn.

# In[22]:


import itertools # construct specialized tools


# In[23]:


def plot_confusion_matrix(cm, classes,normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontsize = 22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 13)
    plt.yticks(tick_marks, classes, fontsize = 13)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 fontsize = 15,
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True', fontsize = 16)
    plt.xlabel('Predicted', fontsize = 16)

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, yhat, labels = [0,1])
np.set_printoptions(precision = 2)


# Plot non-normalized confusion matrix

plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['y=0','y=1'], normalize = False,  title = 'Confusion matrix')
plt.savefig('confusion_matrix.png')

# Note y=1 means that there IS recirculation


# Plotting the confusion matrix using seaborn.

# In[24]:


matrix = confusion_matrix(y_test, yhat)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')


# Conclusion

# There were 28343 records in the dataset, out of which 70% of the data was given for training the model and 30% of the data (8503) were given for testing the model.

# Out of the 8503 data used for testing, 2181 records were misclassified.

# From the Classification Report and the Confusion Matrix, we can infer that y=0 has a high f1 score while y=1 has a low f1 score. This means that the model has a high accuracy when predicting results for y=0 but otherwise for y=1.

# As there is an imbalance in the quantity of results for y=1, to improve the model, oversampling of data where y=1 may help.
