#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import sqlite3


# In[3]:


con = sqlite3.connect(r"C:\Users\sanja\Downloads\password_resources\password_data.sqlite")


# In[4]:


data=pd.read_sql_query("SELECT * FROM Users",con)


# In[5]:


data.shape


# In[6]:


data.head(5)


# In[7]:


data.columns


# In[8]:


data.drop(["index"],axis = 1 ,inplace = True)


# In[9]:


data.duplicated().sum()


# In[10]:


data.isnull().any().sum()


# In[11]:


data.dtypes


# In[12]:


data["strength"].unique()


# In[13]:


data["password"]


# In[14]:


type(data["password"][0])


# In[15]:


data["password"].str.isnumeric()


# In[16]:


data[data["password"].str.isnumeric()]


# In[17]:


data[data["password"].str.isnumeric()].shape


# In[18]:


data[data["password"].str.isupper()]


# In[19]:


data[data["password"].str.isalpha()]


# In[20]:


data[data["password"].str.isalpha()].shape


# In[21]:


data[data["password"].str.isalnum()]


# In[22]:


data[data["password"].str.isalnum()].shape


# In[23]:


data[data["password"].str.istitle()]


# In[24]:


data["password"]


# In[25]:


import string


# In[26]:


string.punctuation


# In[27]:


#function is used when task is repeated


# In[28]:


def find_semantics(row):
    for char in row:
        if char in string.punctuation:
            return 1


# In[29]:


data["password"].apply(find_semantics)


# In[30]:


data[data["password"].apply(find_semantics)==1]


# In[31]:


data[data["password"].apply(find_semantics)==1].sum()


# In[32]:


data[data["password"].apply(find_semantics)==1].apply


# In[33]:


data["password_length"] = data["password"].str.len()


# In[34]:


data["password_length"]


# In[35]:


password = "Pepsi@123"


# In[36]:


[char for char in password if char.islower()]


# In[37]:


len([char for char in password if char.islower()])/len(password)


# In[38]:


def freq_lowercase(row):
    return len([char for char in row if char.islower()])/len(password)


# In[39]:


def freq_uppercase(row):
    return len([char for char in row if char.isupper()])/len(password)


# In[40]:


def freq_numerical_case(row):
    return len([char for char in row if char.isdigit()])/len(password)


# In[41]:


data["freq_lowercase"] = np.round(data["password"].apply(freq_lowercase),3)
data["freq_uppercase"] = np.round(data["password"].apply(freq_uppercase),3)
data["freq_numerical_case"] = np.round(data["password"].apply(freq_numerical_case),3)


# In[42]:


data["freq_lowercase"] 
data["freq_uppercase"]
data["freq_numerical_case"]


# In[43]:


data.head(3)


# In[44]:


def freq_special_case(row):
    special_chars = []
    for char in row:
        if not char.isalpha() and not char.isdigit():
            special_chars.append(char)

    return len(special_chars) 



# In[45]:


data["freq_special_case"] = np.round(data["password"].apply(freq_special_case),3)


# In[46]:


data.head(3)


# In[47]:


data["special_case"] = data["freq_special_case"]/data["password_length"]


# In[48]:


data.head(3)


# In[49]:


data.columns


# In[50]:


data[['password_length' , 'strength']].groupby(['strength']).agg(["min","max","mean","median"])


# In[51]:


cols = ['password_length', 'freq_lowercase',
       'freq_uppercase', 'freq_numerical_case', 'freq_special_case',
       'freq_special_case', 'special_case']
for col in cols:
    print(col)
    print(data[[col , 'strength']].groupby(['strength']).agg(["min","max","mean","median"]))
    print('\n')


# In[52]:


data.columns


# In[53]:


fig ,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(15,7))

sns.boxplot(x = "strength", y = 'password_length', hue="strength" , ax=ax1,data=data)
sns.boxplot(x = "strength", y = 'freq_lowercase' , hue="strength" , ax=ax2,data=data)
sns.boxplot(x = "strength", y = 'freq_uppercase', hue="strength", ax=ax3,data=data)
sns.boxplot(x = "strength", y = 'freq_numerical_case', hue="strength",ax=ax4,data=data)
sns.boxplot(x = "strength", y = 'special_case', hue="strength" ,ax=ax5,data=data)

plt.subplots_adjust(hspace = 0.6)


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns

def get_dist(data, feature):
    fig, axes = plt.subplots(1, 2, figsize=(10,8))

    # Left plot: violin plot
    sns.violinplot(x='strength', y=feature, data=data, ax=axes[0])
    axes[0].set_title(f"Violin Plot of {feature} by Strength")

    # Right plot: KDE distributions for each strength
    sns.kdeplot(data=data[data['strength']==0][feature], color="red", label="0", ax=axes[1])
    sns.kdeplot(data=data[data['strength']==1][feature], color="blue", label="1", ax=axes[1])
    sns.kdeplot(data=data[data['strength']==2][feature], color="yellow", label="2", ax=axes[1])
    axes[1].set_title(f"Distribution of {feature} by Strength")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# In[55]:


data.columns


# In[56]:


get_dist(data ,'freq_lowercase')


# In[57]:


get_dist(data ,'freq_special_case')


# In[58]:


get_dist(data ,'special_case')


# In[59]:


data


# In[60]:


dataframe = data.sample(frac = 1)


# In[61]:


dataframe


# In[62]:


x = list(dataframe["password"])


# In[63]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[64]:


vectorizer = TfidfVectorizer(analyzer = "char")


# In[65]:


X = vectorizer.fit_transform(x)


# In[66]:


X.shape


# In[67]:


X


# In[68]:


X.toarray()


# In[69]:


len(vectorizer.get_feature_names_out())


# In[70]:


df2 = pd.DataFrame(X.toarray() , columns =vectorizer.get_feature_names_out() )


# In[71]:


df2


# In[72]:


data.columns


# In[73]:


df2["password_length"] = data['password_length']
df2["freq_lowercase"] = data['freq_lowercase']


# In[74]:


df2


# In[75]:


y=dataframe["strength"]


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(df2,y,test_size = 0.20,random_state=42)


# In[78]:


X_train.shape


# In[79]:


y_train.shape


# In[80]:


X_test.shape


# In[81]:


y_test.shape


# In[82]:


from sklearn.linear_model import LogisticRegression


# In[83]:


clf = LogisticRegression( multi_class="multinomial")


# In[84]:


clf.fit(X_train,y_train)


# In[85]:


y_pred = clf.predict(X_test)


# In[86]:


y_pred


# In[87]:


from collections import Counter


# In[88]:


Counter(y_pred)


# In[89]:


password = "%@123abcd"


# In[90]:


sample_array = np.array([password])


# In[91]:


sample_matrix = vectorizer.transform(sample_array)


# In[92]:


sample_matrix.toarray()


# In[93]:


sample_matrix.toarray().shape


# In[94]:


password


# In[95]:


len(password)


# In[96]:


[char for char in password if char.islower()]


# In[97]:


len([char for char in password if char.islower()])


# In[98]:


len([char for char in password if char.islower()])/len(password)


# In[99]:


np.append(sample_matrix.toarray() , (9,0.44)).shape


# In[100]:


np.append(sample_matrix.toarray() , (9,0.44)).reshape(1,101)


# In[101]:


np.append(sample_matrix.toarray() , (9,0.44)).reshape(1,101).shape


# In[102]:


new_matrix = np.append(sample_matrix.toarray() , (9,0.44)).reshape(1,101)


# In[103]:


clf.predict(new_matrix)


# In[104]:


def predict():
    password = input("Enter a password : ")
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array)

    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()])/len(password)

    new_matrix2 = np.append(sample_matrix.toarray() ,(length_pass , length_normalised_lowercase)).reshape(1,101)
    result = clf.predict(new_matrix2)

    if result == 0:
        return "Password is weak"
    elif result == 1:
        return "Password is normal"
    else:
        return "Password is strong"


# In[107]:


predict()


# In[ ]:




