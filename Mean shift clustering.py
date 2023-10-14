KINDLY NOTE :THE OUTPUT IS SHARED AS SCREENSHOT IN THE README FILE

# In[50]:


pip install opencv-python


# In[51]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[52]:


img = cv.imread('peppers.jpeg')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()


# In[53]:


img = cv.medianBlur(img, 7)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


# In[54]:


ax = plt.axes(projection ="3d")
ax.scatter3D(img[:,:,0],img[:,:,1],img[:,:,2])
ax.set_title('Pixel Values ')
plt.show()


# In[55]:


img.shape


# In[56]:


X = img.reshape((-1,3))
print("shape: ",X.shape)
print("data type   : ",X.dtype)


# In[57]:


X = np.float32(X)


# In[58]:


bandwidth = estimate_bandwidth(X, quantile=.06, n_samples=3000)
bandwidth 


# In[59]:


ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(X)


# In[60]:


labeled=ms.labels_
labeled


# In[61]:


np.unique(labeled)


# In[62]:


ms.cluster_centers_


# In[63]:


cluster_int8=np.uint8(ms.cluster_centers_)
cluster_int8


# In[64]:


ms.predict(X)


# In[65]:


ax = plt.axes(projection ="3d")
ax.scatter3D(img[:,:,0],img[:,:,1],img[:,:,2])
ax.set_title('Pixel Values ')
plt.show()

ax = plt.axes(projection ="3d")
ax.set_title('Pixel Cluster Values  ')
ax.scatter3D(cluster_int8[:,0],cluster_int8[:,1],cluster_int8[:,2],color='red')
plt.show()


# In[66]:


result=np.zeros(X.shape,dtype=np.uint8)

for label in np.unique(labeled):
    result[labeled==label,:]=cluster_int8[label,:]    
    

result=result.reshape(img.shape)


# In[67]:


plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()


# In[68]:


for label in np.unique(labeled):
    result=np.zeros(X.shape,dtype=np.uint8)
    result[labeled==label,:]=cluster_int8[label,:]  
    plt.imshow(cv.cvtColor(result.reshape(img.shape), cv.COLOR_BGR2RGB))
    plt.show()


# In[69]:


import requests 
url='https://www.plastform.ca/wp-content/themes/plastform/images/slider-image-2.jpg'
name="my_file.jpg"

with open(name, 'wb') as file:
    file.write(requests.get(url, stream=True).content)
    
img = cv.imread(name)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()


# In[76]:


img = cv.medianBlur(img, 7)
X = img.reshape((-1, 3))
X = np.float32(X)

# Estimate bandwidth and apply MeanShift clustering
bandwidth = estimate_bandwidth(X, quantile=0.01, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

# Get cluster labels and cluster centers
labeled = ms.labels_
cluster_int8 = np.uint8(ms.cluster_centers_)
result = np.zeros(X.shape, dtype=np.uint8)

# Create and display segmented images for each cluster
for label in np.unique(labeled):
    result[labeled == label, :] = cluster_int8[label, :]
    result_img = result.reshape(img.shape)
    plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
    plt.show()


# In[77]:


df = pd.read_csv("titanic.csv")
df.head()


# In[78]:


df=df.drop(columns=['Name','Ticket','Cabin','PassengerId','Embarked'])


# In[79]:


df.loc[df['Sex']!='male','Sex']=0
df.loc[df['Sex']=='male','Sex']=1


# In[80]:


df.head()


# In[81]:


df.isna().sum()


# In[82]:


df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[83]:


X=df.drop(columns=['Survived'])


# In[84]:


X=df.apply(lambda x: (x-x.mean())/(x.std()+0.0000001), axis=0)


# In[85]:


X.head()


# In[86]:


bandwidth = estimate_bandwidth(X)
ms = MeanShift(bandwidth=bandwidth , bin_seeding=True)
ms.fit(X)


# In[87]:


X['cluster']=ms.labels_
df['cluster']=ms.labels_


# In[88]:


df.groupby('cluster').mean().sort_values(by=['Survived'], ascending=False)


# In[92]:


def gaussian(d, h):
    return np.exp(-0.5*((d/h))**2) / (h*math.sqrt(2*math.pi))


# In[93]:


s=1 # a sample point
x = np.linspace(-2, 4, num=200)
dist=np.sqrt(((x-s)**2))
kernel_1=gaussian(dist, 1)
kernel_2=gaussian(dist, 3)


# In[94]:


plt.plot(x,kernel_1,label='h=1')
plt.plot(x,kernel_2,label='h=3')
plt.plot(s,0,'x',label="$x_{1}$=1")
plt.hist(s, 10, facecolor='blue', alpha=0.5,label="Histogram")
plt.xlabel('x')
plt.legend()
plt.show()


# In[99]:


S=np.zeros((200))
S[0:100] = np.random.normal(-10, 1, 100)
S[100:200]=np.random.normal(10, 1, 100)
plt.plot(S,np.zeros((200)),'x')
plt.xlabel("$x_{i}$")
plt.show()


# In[100]:


x = np.linspace(S.min()-3, S.max()+3, num=200)
density=kernel_density(S,x)


# In[101]:


plt.plot(x,density,label=" KDE")
plt.plot(S,np.zeros((200,1)),'x',label="$x_{i}$")
plt.xlabel('x')
plt.legend()
plt.show()


# In[102]:


mean_shift=((density.reshape(-1,1)*S).sum(0) / density.sum())-x


# In[103]:


plt.plot(x,density,label=" KDE")
plt.plot(S,np.zeros((200,1)),'x',label="$x_{i}$")
plt.quiver(x, np.zeros((200,1)),mean_shift, np.zeros((200,1)), units='width',label="$m_{h}(x)$")
plt.xlabel('x')
plt.legend()
plt.show()


# In[104]:


Xhat=np.copy(S.reshape(-1,1))
S_=S.reshape(-1,1)


for k in range(3):
    plt.plot(x,density,label=" KDE")
    plt.plot(Xhat,np.zeros((200,1)),'x',label="$\hat{x}^{k}_i$,k="+str(k))
    plt.xlabel('x')
    plt.legend()
    plt.show()
  
    for i,xhat in enumerate(Xhat):
        dist=np.sqrt(((xhat-S_)**2).sum(1))
        weight = gaussian(dist, 2.5)
        Xhat[i] = (weight.reshape(-1,1)*S_).sum(0) / weight.sum()


# In[105]:


np.unique(Xhat.astype(int))

