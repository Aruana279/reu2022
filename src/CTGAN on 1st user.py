#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sdv.demo import load_tabular_demo
import pandas as pd
import numpy as np
import random
import csv
from sdv.demo import load_tabular_demo
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
# import shap # v0.39.0
# shap.initjs()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, 
                              RandomForestClassifier)


import xgboost as xgb


# In[16]:


data = pd.read_csv('3.features_labels.csv')
data.head()


# In[17]:


data=data.fillna(0)
data

# y=data.filter(regex='label:LYING_DOWN')
# y.head()


# In[4]:


discrete=[]
for c in data.columns:
    if "discrete:" in c:
        discrete.append(c)

print(len(discrete))


labels=[]
for c in data.columns:
    if "label:" in c:
        labels.append(c)
print(len(labels))



continuous=[]
for c in data.columns:
    if "label:" not in c and "discrete:" not in c:
        continuous.append(c)
continuous.remove('timestamp')
continuous.remove('label_source')
print(len(continuous))


# In[6]:


label=labels[0]
temp=[i for i in labels if i != label]
print(temp)
print(label)


# In[19]:


make_graphs=False
results={}
for label in labels:
    print(label)
    temp=[i for i in labels if i != label]
    y=data[label]
    x=data[continuous+discrete+temp]
    feature_names = continuous+discrete+temp
#     forest = RandomForestClassifier(random_state=0)
#     forest.fit(x, y)
#     std = np.std([tree.feature_importances_ for tree in xgb_model.estimators_], axis=0)
#     importances = forest_model.feature_importances_
#     forest_importances = pd.Series(importances, index=feature_names)
#     info=forest_importances.sort_values(ascending=False)[:10].to_dict()
#     results[label]=info
    xgb_model = xgb.XGBRegressor(random_state=0)
    xgb_model.fit(x, y)
    importances = xgb_model.feature_importances_
    xgb_importances = pd.Series(importances, index=feature_names)
    info=xgb_importances.sort_values(ascending=False)[:10].to_dict()
    results[label]=info
    print(xgb_importances.sort_values(ascending=False)[:10])
    
    if make_graphs: 
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
print(xgb_importances.sort_values(ascending=False)[:10])
print("\n")
# for key, value in results.items():
#         print(key, ' : ', value)
        
# import json
# with open("57.json", "w") as fp:
#     json.dump(results, fp, indent=4) 


# In[18]:


# make_graphs=False
# results={}
# for label in labels:
#     print(label)
#     temp=[i for i in labels if i != label]
#     y=data[label]
#     x=data[continuous+discrete+temp]
#     feature_names = continuous+discrete+temp
#     forest = RandomForestClassifier(random_state=0)
#     forest.fit(x, y)
#     std = np.std([tree.feature_importances_ for tree in xgb_model.estimators_], axis=0)
#     importances = forest_model.feature_importances_
#     forest_importances = pd.Series(importances, index=feature_names)
#     info=forest_importances.sort_values(ascending=False)[:10].to_dict()
#     results[label]=info
#     
#     results[label]=info
#     
#     if make_graphs: 
#         fig, ax = plt.subplots()
#         forest_importances.plot.bar(yerr=std, ax=ax)
#         ax.set_title("Feature importances using MDI")
#         ax.set_ylabel("Mean decrease in impurity")
#         fig.tight_layout()
#         plt.show()
#print(forest_importances.sort_values(ascending=False)[:10])
# for key, value in results.items():
#         print(key, ' : ', value)
        
# import json
# with open("57.json", "w") as fp:
#     json.dump(results, fp, indent=4) 


# In[ ]:





# In[ ]:


# Printing only one plot

# feature_names = continuous+discrete+temp
# forest = RandomForestClassifier(random_state=0)
# forest.fit(x, y)

# importances = forest.feature_importances_
# forest_importances = pd.Series(importances, index=feature_names)
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()

# plt.show()


# In[18]:


from ctgan import CTGANSynthesizer

model = CTGANSynthesizer(epochs=2)

model.fit(data, discrete)


# In[6]:


new_data = model.sample(10)


# 

# In[7]:


new_data.head()


# In[8]:


synthetic_data=model.sample(len(data))


# In[9]:


model.save('extra_sensory_ctgan.pkl')


# In[10]:


loaded = CTGANSynthesizer.load('extra_sensory_ctgan.pkl')
new_data[new_data.timestamp == new_data.timestamp.value_counts().index[0]]


# In[11]:


#EVALUATION


# In[14]:


from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula
real_data =  pd.read_csv('3.features_labels.csv')
model = GaussianCopula()
model.fit(real_data)
synthetic_data = model.sample(len(real_data))


# In[15]:


real_data.head()


# In[16]:


synthetic_data.head()


# In[12]:


from sdv.evaluation import evaluate
evaluate(synthetic_data, data)


# In[14]:


evaluate(synthetic_data, data, aggregate=False)


# In[ ]:


evaluate(synthetic_data, data, metrics=['F-1 test'])


# In[ ]:


def evaluation(model, test_loader):
    softmax = nn.Softmax(dim = 1)
    for batch in test_loader: #Runs once since the batch is the entire testing data
        features, labels = batch
        _, preds = torch.max(softmax(model(features.float())), dim = 1) #Getting the model's predictions
        report = metrics.classification_report(labels, preds, digits = 3, output_dict = True, zero_division = 0)
        f1_score = pd.DataFrame(report).transpose().loc['weighted avg', :]['f1-score']
    return f1_score


# In[ ]:


y.head()
data.head()


# In[ ]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# X, y = make_classification(
#     n_samples=1000,
#     n_features=len(discrete),
#     n_informative=len(labels),
#     n_redundant=0,
#     n_repeated=0,
#     n_classes=len(labels),
#     random_state=0,
#     shuffle=False,
    
    
#     n_samples=1000,
#     n_features=len(data.column),
#     n_informative=len(labels),
#     n_redundant=0,
#     n_repeated=0,
#     n_classes=len(labels),
#     random_state=0,
#     shuffle=False,
# )

# print(X.shape, y.shape)

X_train,y_train = train_test_split(data, y)


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier

# feature_names = [f"feature {i}" for i in range(X.shape[1])]
# forest = RandomForestClassifier(random_state=0)
# forest.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
feature_names = labels # e.g. ['A', 'B', 'C', 'D', 'E']
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)


# In[ ]:


import time
import numpy as np

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(len(importances))
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[ ]:


import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)
print(len(feature_names))
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# In[ ]:





# In[ ]:





# In[ ]:


# from sklearn.inspection import permutation_importance

# start_time = time.time()
# result = permutation_importance(
#     forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
# elapsed_time = time.time() - start_time
# print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# forest_importances = pd.Series(result.importances_mean, index=feature_names)


# In[ ]:


# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.show()


# In[ ]:


Linear Regression Feature Importance


# In[ ]:



# test regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)


# In[ ]:


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=len(discrete), n_informative=len(discrete), random_state=0)
# define the model
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0s, Score: %.5f' % (discrete[i],v))
# plot feature importance
linear_importance=pd.Series(importance, index=discrete)
# pyplot.bar([x for x in range(len(importance))], linear_importance)
pyplot.bar(discrete, linear_importance)
pyplot.suptitle("Linear Regression Feature Importance")
pyplot.ylabel("Crude feature importance score")
pyplot.show()


# In[ ]:


#shap score


# In[ ]:


# #Logistic Regression Feature Importance

# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from matplotlib import pyplot
# # define dataset
# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# # define the model
# model = LogisticRegression()
# # fit the model
# model.fit(X, y)
# # get importance
# importance = model.coef_[0]
# # summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()


# In[ ]:


data = pd.read_csv('00EABED2-271D-49D8-B599-1D4A09240601.features_labels.csv')
data.head()
data=data.fillna(0)
data

discrete=[]
for c in data.columns:
    if "discrete:" in c:
        discrete.append(c)

print(len(discrete))


labels=[]
for c in data.columns:
    if "label:" in c:
        labels.append(c)
print(len(labels))



continuous=[]
for c in data.columns:
    if "label:" not in c and "discrete:" not in c:
        continuous.append(c)
continuous.remove('timestamp')
continuous.remove('label_source')
print(len(continuous))

label=labels[0]
temp=[i for i in labels if i != label]



# In[ ]:


make_graphs=False
for label in labels:
    print(label)
    temp=[i for i in labels if i != label]
    np.set_printoptions(formatter={'float':lambda x:"{:.4f}".format(x)})
    pd.options.display.float_format = "{:.3f}".format

    sns.set(style='darkgrid', context='talk', palette='rainbow')
    y=data[label]
    x=data[continuous+discrete+temp]
    feature_names = continuous+discrete+temp
    
    
#     model = Sequential()
#     model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#     model.fit(x, y)



    model = RandomForestRegressor(random_state=None)
    model.fit(x, y)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    importances = model.feature_importances_
    model_importances = pd.Series(importances, index=feature_names)
    explainer = shap.Explainer(model)
    shap_test = explainer(x)
    print(f"Shap values length: {len(shap_test)}\n")
    print(f"Sample shap value:\n{shap_test[0]}")
    if make_graphs: 
        fig, ax = plt.subplots()
        model_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using shap score")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
        
    print(shap_importances.sort_values(ascending=False)[:10])
    print()


# In[ ]:


make_graphs=False
for label in labels:
    print(label)
    temp=[i for i in labels if i != label]
    np.set_printoptions(formatter={'float':lambda x:"{:.4f}".format(x)})
    pd.options.display.float_format = "{:.3f}".format

    sns.set(style='darkgrid', context='talk', palette='rainbow')
    y=data[label]
    x=data[continuous+discrete+temp]
    feature_names_shap = continuous+discrete+temp
    
    
    model = Sequential()
    model.add(Dense(32, input_dim=x.shape[1], activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(x, y)
    
    explainer = shap.DeepExplainer(model, x)
    shap_values = explainer.shap_values(x)
    shap.summary_plot(shap_values[0], plot_type = 'bar', feature_names = feature_names_shap)



#     model = RandomForestRegressor(random_state=None)
#     model.fit(x, y)
#     std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
#     importances = model.feature_importances_
#     model_importances = pd.Series(importances, index=feature_names)
#     explainer = shap.Explainer(model)
#     shap_test = explainer(x)
#     print(f"Shap values length: {len(shap_test)}\n")
#     print(f"Sample shap value:\n{shap_test[0]}")
    if make_graphs: 
        fig, ax = plt.subplots()
        model_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using shap score")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
        
    print(shap_importances.sort_values(ascending=False)[:10])
    print()



# In[ ]:




