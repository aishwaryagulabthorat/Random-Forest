#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the csv file 
df = pd.read_csv('/Users/aishwaryathorat/Movies/MS Courses/Upg/Random Forest/heart_v2.csv')


# In[3]:


df.columns


# In[4]:


df.head()


# In[5]:


# Putting feature variable to X
X = df.drop('heart disease',axis=1)

# Putting response variable to y
y = df['heart disease']


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


# Fitting the decision tree with default hyperparameters, apart from max_depth which is 3 so that we can plot and read the tree.

# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)


# In[10]:


get_ipython().system('pip install six')


# In[11]:


# Importing required packages for visualization
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz


# In[12]:


# plotting tree with max_depth=3
dot_data = StringIO()  

export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                feature_names=X.columns, 
                class_names=['No Disease', "Disease"])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#Image(graph.create_png(),width=800,height=900)
#graph.write_pdf("dt_heartdisease.pdf")


# #### Evaluating model performance

# In[13]:


y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)


# In[14]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[15]:


print(accuracy_score(y_train, y_train_pred))
confusion_matrix(y_train, y_train_pred)


# In[16]:


print(accuracy_score(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)


# Creating helper functions to evaluate model performance and help plot the decision tree

# In[17]:


def get_dt_graph(dt_classifier):
    dot_data = StringIO()
    export_graphviz(dt_classifier, out_file=dot_data, filled=True,rounded=True,
                    feature_names=X.columns, 
                    class_names=['Disease', "No Disease"])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph


# In[18]:


def evaluate_model(dt_classifier):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))


# ### Without setting any hyper-parameters

# In[19]:


dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)


# In[20]:


gph = get_dt_graph(dt_default)
Image(gph.create_png())


# In[21]:


evaluate_model(dt_default)


# ### Controlling the depth of the tree

# In[22]:


get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# In[23]:


dt_depth = DecisionTreeClassifier(max_depth=3)
dt_depth.fit(X_train, y_train)


# In[24]:


gph = get_dt_graph(dt_depth) 
Image(gph.create_png())


# In[25]:


evaluate_model(dt_depth)


# ### Specifying minimum samples before split

# In[26]:


dt_min_split = DecisionTreeClassifier(min_samples_split=20)
dt_min_split.fit(X_train, y_train)


# In[27]:


gph = get_dt_graph(dt_min_split) 
Image(gph.create_png())


# In[28]:


evaluate_model(dt_min_split)


# ### Specifying minimum samples in leaf node

# In[29]:


dt_min_leaf = DecisionTreeClassifier(min_samples_leaf=20, random_state=42)
dt_min_leaf.fit(X_train, y_train)


# In[30]:


gph = get_dt_graph(dt_min_leaf)
Image(gph.create_png())


# In[31]:


evaluate_model(dt_min_leaf)


# ### Using Entropy instead of Gini

# In[32]:


dt_min_leaf_entropy = DecisionTreeClassifier(min_samples_leaf=20, random_state=42, criterion="entropy")
dt_min_leaf_entropy.fit(X_train, y_train)


# In[33]:


gph = get_dt_graph(dt_min_leaf_entropy)
Image(gph.create_png())


# In[34]:


evaluate_model(dt_min_leaf_entropy)


# ### Hyper-parameter tuning

# In[35]:


dt = DecisionTreeClassifier(random_state=42)


# In[36]:


from sklearn.model_selection import GridSearchCV


# In[37]:


# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[37]:


# #grid_search = GridSearchCV(estimator=dt, 
# #                          param_grid=params, 
# #                           cv=4, n_jobs=-1, verbose=1, scoring = "f1")


# In[38]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[39]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)\n')


# In[40]:


score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()


# In[42]:


score_df.nlargest(5,"mean_test_score")


# In[43]:


grid_search.best_estimator_


# In[44]:


dt_best = grid_search.best_estimator_


# In[45]:


evaluate_model(dt_best)


# to print the detailed evaluation of the model 

# Precision: The ratio of correctly predicted positive observations to the total predicted positives. 
# It is also known as the Positive Predictive Value.
# 
# Precision = TP / (TP + FP)
# TP: True Positives
# FP: False Positives
# 
# 
# Recall: The ratio of correctly predicted positive observations to the all observations in the actual class. 
# It is also known as Sensitivity or True Positive Rate.
# 
# Recall = TP / (TP + FN)
# FN: False Negatives
# 
# 
# F1-Score: The weighted average of Precision and Recall. 
# The F1 Score is more useful than accuracy, especially if you have an uneven class distribution. 
# It is the harmonic mean of Precision and Recall.
# 
# F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
# Support: The number of actual occurrences of each class in the dataset.

# In[48]:


from sklearn.metrics import classification_report


# In[49]:


print(classification_report(y_test, dt_best.predict(X_test)))


# In[50]:


gph = get_dt_graph(dt_best)
Image(gph.create_png())


# ## Random Forest

# In[51]:


from sklearn.ensemble import RandomForestClassifier


# Going to create 10 trees with max depth of 3

# In[59]:


rf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=3,oob_score=True)


# In[61]:


rf.fit(X_train, y_train)


# To get all the n trees with their hyperparameters

# In[54]:


rf.estimators_[0]


# In[62]:


rf.oob_score_


# extracting 5th tree

# In[63]:


sample_tree = rf.estimators_[4]


# In[64]:


gph = get_dt_graph(sample_tree)
Image(gph.create_png(), width=700, height=700)


# extracting and printing third tree

# In[65]:


gph = get_dt_graph(rf.estimators_[2])
Image(gph.create_png(), width=700, height=700)


# In[66]:


evaluate_model(rf)


# #### Grid search for hyper-parameter tuning

# In[67]:


classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1)


# In[68]:


# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [1, 2, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'max_features': [2,3,4],
    'n_estimators': [10, 30, 50, 100, 200]
}


# CV=4 creating 4 foldes of the training data, 1 validation and other 3 training datasets

# In[69]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator=classifier_rf, param_grid=params, 
                          cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[70]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X,y)\n')


# best combination of parameters which will give highe accuracy

# In[71]:


rf_best = grid_search.best_estimator_


# In[72]:


rf_best


# In[73]:


evaluate_model(rf_best)


# In[74]:


sample_tree = rf_best.estimators_[0]


# In[75]:


gph = get_dt_graph(sample_tree)
Image(gph.create_png())


# In[76]:


gph = get_dt_graph(rf_best.estimators_[0])
Image(gph.create_png(), height=600, width=600)


# In[77]:


gph = get_dt_graph(rf_best.estimators_[10])
Image(gph.create_png(), height=600, width=600)


# ### Variable importance in RandomForest and Decision trees

# In[78]:


classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)


# In[79]:


classifier_rf.fit(X_train, y_train)


# In[80]:


classifier_rf.feature_importances_


# In[81]:


imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": classifier_rf.feature_importances_
})


# In[82]:


imp_df.sort_values(by="Imp", ascending=False)


# In[ ]:




