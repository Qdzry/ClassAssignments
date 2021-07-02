#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# In[9]:


# 使用numpy里的linspace函数产生题目要求范围内的均匀递增数组
## 注：由于题目给定的u取值范围并不能把此次傅里叶变换后所有的峰值体现出来，故扩大了范围
x=np.linspace(0,2*np.pi,200)
u=np.linspace(0,40,150)


# In[10]:


# 定义原函数f(x),输入x返回f(x)
def f_x(x):
    return(30*np.sin(x)+20*np.cos(10*x)+10*np.sin(20*x)+np.cos(30*x))


# In[11]:


# 定义离散函数傅里叶变换的函数
## 注：对题目给定公式作了一定的修改，把指数里的2pi/N放到指数外了（否则不出现峰值）
# 输入均匀递增的x和对应的f(x)的数组，以及u值，得出傅里叶变换F(u)的值
def fourier_tran(x,Y,u):
    N=x.size
    F_u=0
    # 依题意在一个周期内进行累加
    for i in range(N):
        F_u+=Y[i]*np.exp(-(0+1j)*x[i]*u)*2*np.pi/N
    return abs(F_u)#结果是复数，取其模


# In[12]:


Y=f_x(x) # Y用来存放对应x的原函数的值
F=fourier_tran(x,Y,u) # F用来存放对应不同u的傅里叶变换的值


# In[13]:


#创建图像，分为ax1,ax2上下两个子图
fig=plt.figure()
ax1=plt.axes([0.1,0.7,1,0.45])
ax2=plt.axes([0.1,0.1,1,0.45])
ax1.plot(x,Y,color='blue',label='f(x)')
ax2.plot(u,F,color='red',label='|F(u)|')
#加入轴标签和图例
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax2.set_xlabel("u")
ax2.set_ylabel("|F(u)|")
ax1.legend()
ax2.legend()
#画出此图（其实不加也能画出）
plt.show()


# In[14]:


plt.close(fig)#关闭图片


# In[ ]:




