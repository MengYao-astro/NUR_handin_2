#!/usr/bin/env python
# coding: utf-8



# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.ioff()


# ### Read the data file and label the data

# In[2]:


'''
Question 6 : Classifying gamma-ray bursts
'''
print('Question 6 : Classifying gamma-ray bursts')
#data = pd.read_csv('GRBs.txt', sep='\s+')
data = np.genfromtxt('GRBs.txt', usecols=(2,3,4,5,6,7,8))
data = pd.DataFrame(data, columns=['Redshift', 'T90', 'log(M*/M_sun)', 'SFR',
                                   'log(Z/Z_sun)', 'SSFR', 'AV'])
data['label'] = None
#Assign the label based on T90 parameter
data['label'][data['T90'] < 10] = 0
data['label'][data['T90'] >= 10] = 1
data['label'] = data['label'].convert_objects(convert_numeric=True)
print('Data Shape:', data.shape)


# ### Histogram the features.
# I first plot the histogram to check the feature values.

# In[3]:


# Check the missing data
index_M = data['log(M*/M_sun)'] != -1
index_SFR = data['SFR'] != -1
index_Z = data['log(Z/Z_sun)'] != -1
index_SSFR = data['SSFR'] != -1
index_AV = data['AV'] != -1

fig = plt.figure(figsize=(10,6))

ax1 = fig.add_subplot(2,3,1)
M = ax1.hist(data['log(M*/M_sun)'][index_M], density=True)
ax1.set_xlabel('log(M*/M_sun) %s '%len(data['log(M*/M_sun)'][index_M]))

ax2 = fig.add_subplot(2,3,2)
SFR = ax2.hist(data['SFR'][index_SFR], density=True)
ax2.set_xlabel('SFR %s' %len(data['SFR'][index_SFR]))

ax3 = fig.add_subplot(2,3,3)
Z = ax3.hist(data['log(Z/Z_sun)'][index_Z], density=True)
ax3.set_xlabel('log(Z/Z_sun %s)' %len(data['log(Z/Z_sun)'][index_Z]))

ax4 = fig.add_subplot(2,3,4)
SSFR = ax4.hist(data['SSFR'][index_SSFR], density=True)
ax4.set_xlabel('SSFR %s' %len(data['SSFR'][index_SSFR]))

ax5 = fig.add_subplot(2,3,5)
AV = ax5.hist(data['AV'][index_AV], density=True)
ax5.set_xlabel('AV %s' %len(data['AV'][index_AV]))

ax6 = fig.add_subplot(2,3,6)
ax6.hist(data['Redshift'])
ax6.set_xlabel('Redshift')
fig.savefig('q6_1.png')


# It's seems that the mass M and metality Z conform to the gauss distribution. Meanwhile, SFR fits the exoponential distribution. Therefore, I choose to fill the non-determinded variables with random number with specific distribution. 
# The data with SSFR and AV are few, so I decided to give up these two features.

# In[4]:


#Processing the missing data
miu_M = np.mean(data['log(M*/M_sun)'][index_M])
sigma_M = np.std(data['log(M*/M_sun)'][index_M])
lambda_SFR = np.mean(data['SFR'][index_SFR])
miu_Z = np.mean(data['log(Z/Z_sun)'][index_Z])
sigma_Z = np.std(data['log(Z/Z_sun)'][index_Z])

index = data['log(M*/M_sun)'] == -1
index_len = len(index)
data['log(M*/M_sun)'][index] = np.random.normal(miu_M, sigma_M, index_len)   

index = data['log(Z/Z_sun)'] == -1
index_len = len(index)
data['log(Z/Z_sun)'][index] = np.random.normal(miu_Z, sigma_Z, size=index_len)   

index = data['SFR'] == -1
index_len = len(index)
data['SFR'][index] = np.random.exponential(lambda_SFR, index_len)   


# ## Part 2, Train the classification applying the gradiant ascend algorithm
# I plot two figures to show my results.

# In[5]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def binary_crossentropy(y_true, y_predict):
#     m = y_true.shape[0]
#     return -1/m * ()

def load_data(data):
    cols = ['Redshift', 'log(M*/M_sun)', 'SFR', 'log(Z/Z_sun)']
    data_Input = pd.DataFrame(data, columns=cols)
    data_Input = np.array(data_Input)
    data_Label = data['label']
    data_Label = np.array(data_Label)
    data_Input = np.insert(data_Input, 0, 1, axis=1)
    return data_Input, data_Label

def grad_ascent(data_Input, data_Label, alpha, epochs, loss =False):
    data_Mat = np.mat(data_Input)
    label_Mat = np.mat(data_Label).transpose()
    m, n = np.shape(data_Mat)
    weights = np.random.normal(0.5,0.2,(n,1))
    Loss = []
    for i in range(epochs):
        h = sigmoid(data_Mat * weights)
        weights = weights + alpha * data_Mat.transpose() * (h - label_Mat) / m
        if loss == True:
            if i % 2 == 0:
                Loss.append(-np.sum((np.array(label_Mat) - np.array(h))**2) / (2*m))

    return weights, Loss


# In[6]:


data_in, data_lab = load_data(data)
epoch = 1300
W, loss = grad_ascent(data_in, data_lab, alpha=0.001, epochs=epoch, loss=True)

W = np.array(W)
z = np.dot(data_in, W)
z = z.astype(float)
Prediction = sigmoid(z)
Prediction = np.array(Prediction, dtype=int)

fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(121)
ax.hist(Prediction, label='Predict',align='left')
ax.hist(data['label'], label='True Class')#, align='right')
ax.legend(loc = 2)
acc = np.sum(data['label'] == Prediction.flatten()) / data.shape[0]
ax.text(0, 150, 'Accuracy : %f' %acc)

ax2 = fig.add_subplot(122)
ax2.plot(np.arange(0,epoch,2),loss)
fig.savefig('q6_2.png')



