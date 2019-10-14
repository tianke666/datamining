import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#评价指标
from sklearn import metrics

#各种聚类方法
#Kmeans、亲和力传播、均值迁移、谱聚类、层次聚类（ward)、凝聚聚类、DBSCAN、高斯混合模型
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.datasets import load_digits
#PCA用于降维
from sklearn.decomposition import PCA

#标准化：去均值，方差规模化
# 创建一组特征数据，每一行表示一个样本，每一列表示一个特征
# Standardization标准化:将特征数据的分布调整成标准正太分布，也叫高斯分布，也就是使得数据的均值维0，方差为1.
# 标准化的原因在于如果有些特征的方差过大，则会主导目标函数从而使参数估计器无法正确地去学习其他特征。
# 标准化的过程为两步：去均值的中心化（均值变为0）；方差的规模化（方差变为1）。
# 在sklearn.preprocessing中提供了一个scale的方法，可以实现以上功能。
#标准化是针对每一列的
from sklearn.preprocessing import scale
from time import time

from plot_kmeans_digits import n_digits

digits=load_digits()
#获得原始数据
origin_data=digits.data
#获得原始数据的标签，即属于哪一类
labels=digits.target
#查看数据形状与标签个数是否一致
print(origin_data.shape,labels.shape)
#对原始数据进行标准化
data=scale(origin_data)
#查看label中一共有多少类
n_classes=len(np.unique(labels))
print(n_classes)

km=KMeans(init='random',n_clusters=10)
ap=AffinityPropagation()
ms=MeanShift()
sc=SpectralClustering(n_clusters=n_digits, gamma=0.1)
ac=AgglomerativeClustering(n_clusters=n_digits, linkage='average')
db=DBSCAN()
gm=GaussianMixture(n_components=10)

km.fit(data)
ap.fit(data)
ms.fit(data)
sc.fit(data)
ac.fit(data)
db.fit(data)
gm.fit(data)

t0=time()
km.fit(data)
print('name\t\ttime\t\th_score\t\tc_score\t\tnmi')
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('K-Means',time()-t0,metrics.homogeneity_score(labels,km.labels_),metrics.completeness_score(labels, km.labels_), metrics.normalized_mutual_info_score(labels,km.labels_)))
t0=time()
ap.fit(data)
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('AP',time()-t0,metrics.homogeneity_score(labels,ap.labels_),metrics.completeness_score(labels, ap.labels_), metrics.normalized_mutual_info_score(labels,ap.labels_)))
t0=time()
ms.fit(data)
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('Mean-Shift',time()-t0,metrics.homogeneity_score(labels,ms.labels_),metrics.completeness_score(labels, ms.labels_), metrics.normalized_mutual_info_score(labels,ms.labels_)))
#这里数据集因为不满足谱聚类的某些条件执行时间较长
#t0=time()
sc.fit(data)
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('SC',time()-t0,metrics.homogeneity_score(labels,sc.labels_),metrics.completeness_score(labels, sc.labels_), metrics.normalized_mutual_info_score(labels,sc.labels_)))
t0=time()
ac.fit(data)
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('AC',time()-t0,metrics.homogeneity_score(labels,ac.labels_),metrics.completeness_score(labels, ac.labels_), metrics.normalized_mutual_info_score(labels,ac.labels_)))
t0=time()
db.fit(data)
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('DBSCAN',time()-t0,metrics.homogeneity_score(labels,db.labels_),metrics.completeness_score(labels, db.labels_), metrics.normalized_mutual_info_score(labels,db.labels_)))
t0=time()
gm.fit(data)
print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %('GM',time()-t0,metrics.homogeneity_score(labels,gm.predict(data)),metrics.completeness_score(labels, gm.predict(data)), metrics.normalized_mutual_info_score(labels,gm.predict(data))))