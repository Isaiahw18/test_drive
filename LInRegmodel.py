#!/usr/bin/env python
# coding: utf-8

# # linearRegression()

# In[3]:


# Import Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st



# In[6]:


## import the data!
datafile = pd.read_excel(r'Salary_Data.xlsx')
datafile


# In[7]:


X = datafile.iloc[:, :-1].values
y = datafile.iloc[:, -1].values
# seperates the two columns into their own personal array, X is years in career ,y is salary 


# In[9]:


X, y


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[12]:


regressor


# In[16]:


y_pred = regressor.predict(X_test)
y_pred


# In[17]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[18]:


# the Line passes through a couple data points , i dont think this data was linear for the training data.


# In[19]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:
st.title('Salary Estimate Predictor')
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.header('Enter Experience in years:')
st.number_input("Enter Int Value:", min_value:0.1, max_value:12)
salary = predict(datafile)
st.success('f'your predicted salary total is ${salary[0]:.2f} USD')')

