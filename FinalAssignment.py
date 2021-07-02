#!/usr/bin/env python
# coding: utf-8

# # 期末作业

# #### 期末心得：
# 这个学期学了机器学习的一些基本概念，四大类问题（包括分类、回归、聚类和降维）和一些经典算法（knn，逻辑回归，支持向量机，决策树，一些集成算法如随机森林、AdaBoost、梯度提升法，朴素贝叶斯，k-means 聚类，谱聚类等）,对这个学期的学习总体还是满意的。知道了这些算法，希望能在接下来的学习科研中派上一些用场吧，比如在处理一些难以解析化表示的数据的时候，但感觉还是要在实践中继续学习呀。

# ## 1.分类算法的比较

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
from sklearn import svm, model_selection, neural_network, neighbors, tree, linear_model, ensemble, naive_bayes, datasets


# In[2]:


data = datasets.load_iris()
X_iris, y_iris = data.data, data.target

data = datasets.load_digits()
X_digits, y_digits = data.data, data.target

X_blobs, y_blobs = datasets.make_blobs(n_samples=500, centers=3, random_state=0, cluster_std=2)

X_circles, y_circles = datasets.make_circles(n_samples=500, factor=0.5, noise=0.2)

X_moons, y_moons = datasets.make_moons(n_samples=500, noise=0.1)


# In[3]:


def compare_and_draw(X, y, rlts=[], labels=[], title='',cv=5):
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_subplot(111)
    for i, rlt in enumerate(rlts):
        xs, ys = [i+row*0.05 for row in range(len(rlt))], rlt
        ax.scatter(xs, ys, label = labels[i])
        print(labels[i], '\n\t', mean(rlt), '\t', rlt)
    ax.set_xticks(range(len(rlts)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('accuracy score', fontsize=22)
    fig.suptitle(title, fontsize=24)
    plt.show()


# In[7]:


for X, y, dataname in[
    [X_iris, y_iris, 'iris'],
    [X_digits, y_digits, 'digits'],
    [X_blobs, y_blobs, 'blobs'],
    [X_moons, y_moons, 'moons'],
    [X_circles, y_circles, 'circles']
    ]:
    print(dataname)
    rlts=[]
    labels=[]
    for model, modelname in [
        [neighbors.KNeighborsClassifier(n_neighbors=3), 'knn_3'],
        [neighbors.KNeighborsClassifier(n_neighbors=5), 'knn_5'],
        [neighbors.KNeighborsClassifier(n_neighbors=10), 'knn_10'],
        [linear_model.LogisticRegressionCV(), 'logistic'],
        [neural_network.MLPClassifier(hidden_layer_sizes=100), 'neural_100'],
        [neural_network.MLPClassifier(hidden_layer_sizes=200), 'neural_200'],
        [svm.SVC(kernel='linear'), 'SVM_linear'],
        [svm.SVC(kernel='rbf'), 'SVM_linear'],
        [tree.DecisionTreeClassifier(), 'DecisionTree'],
        [ensemble.RandomForestClassifier(), 'RandomForest'],
        [ensemble.AdaBoostClassifier(algorithm='SAMME.R', learning_rate=2), 'AdaBoost'],
        [ensemble.GradientBoostingClassifier(), 'GBDT'],
        [naive_bayes.GaussianNB(),'naive_bayes']
        ]:
        rlt=model_selection.cross_val_score(model, X, y, cv=5)
        rlts.append([xx for xx in rlt]); labels.append(modelname)
        print(modelname)
        print(rlt)
    compare_and_draw(X,y,rlts,labels)


# ## 2.神经网络(week16)

# In[12]:


X, y=datasets.make_moons(500, noise=0.15, random_state=1)
scores = []
layer_sets = []
for n in [20,50,100,500,1000]:
    model = neural_network.MLPClassifier(hidden_layer_sizes=(n,))
    model.fit(X, y)
    y_predict = model.predict(X)
    layer_sets.append(n)
    score = model_selection.cross_val_score(model, X, y, cv=5)
    scores.append(score)

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(range(len(scores)), [xx.mean() for xx in scores], marker='p')
ax.set_xticks(range(len(scores)))
ax.set_xticklabels([str(xx) for xx in layer_sets], rotation=30);
ax.set_xlabel('hidden layer size', fontsize=15)
ax.set_ylabel('cross validation score', fontsize=15)
ax.grid()


# In[ ]:




