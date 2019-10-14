# 实验一： Clustering with sklearn 

##### 学院：计算机科学与技术学院    　　　 姓名：田可 　　 学号：201944765   　　　  时间：2019/10/14



## 一、实验项目名称：

​			Clustering with sklean

## 二、实验目的：

​			1、掌握聚类算法的种类；

​			2、掌握不同聚类算法的特点；

​			3、测试sklearn中以下不同算法在给定两个数据集上的聚类效果。

## 三、实验环境：

​			Python3、Windows10、OpenCV、Numpy等



## 四、实验步骤：

### 1、找到相关聚类算法及其特性：

​						

| Method name                                                  | Parameters                                                   | Scalability                                                  | Usecase                                                      | Geometry (metric used)                       |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :------------------------------------------- |
| [K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means) | number of clusters                                           | Very large `n_samples`, medium `n_clusters` with [MiniBatch code](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans) | General-purpose, even cluster size, flat geometry, not too many clusters | Distances between points                     |
| [Affinity propagation](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation) | damping, sample preference                                   | Not scalable with n_samples                                  | Many clusters, uneven cluster size, non-flat geometry        | Graph distance (e.g. nearest-neighbor graph) |
| [Mean-shift](https://scikit-learn.org/stable/modules/clustering.html#mean-shift) | bandwidth                                                    | Not scalable with `n_samples`                                | Many clusters, uneven cluster size, non-flat geometry        | Distances between points                     |
| [Spectral clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) | number of clusters                                           | Medium `n_samples`, small `n_clusters`                       | Few clusters, even cluster size, non-flat geometry           | Graph distance (e.g. nearest-neighbor graph) |
| [Ward hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | number of clusters or distance threshold                     | Large `n_samples` and `n_clusters`                           | Many clusters, possibly connectivity constraints             | Distances between points                     |
| [Agglomerative clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | number of clusters or distance threshold, linkage type, distance | Large `n_samples` and `n_clusters`                           | Many clusters, possibly connectivity constraints, non Euclidean distances | Any pairwise distance                        |
| [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) | neighborhood size                                            | Very large `n_samples`, medium `n_clusters`                  | Non-flat geometry, uneven cluster sizes                      | Distances between nearest points             |
| [OPTICS](https://scikit-learn.org/stable/modules/clustering.html#optics) | minimum cluster membership                                   | Very large `n_samples`, large `n_clusters`                   | Non-flat geometry, uneven cluster sizes, variable cluster density | Distances between points                     |
| [Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html#mixture) | many                                                         | Not scalable                                                 | Flat geometry, good for density estimation                   | Mahalanobis distances to centers             |
| [Birch](https://scikit-learn.org/stable/modules/clustering.html#birch) | branching factor, threshold, optional global clusterer.      | Large `n_clusters` and `n_samples`                           | Large dataset, outlier removal, data reduction.              | Euclidean distance between points            |

### 2、用以上算法对数据集进行测试：

​         具体代码详见 https://github.com/yihao-guo/Homework/tree/master/Homework1 

## 五、实验总结：

​		本次作业实验主要目的是了解不同聚类算法，不同聚类算法的特点以及具体的使用情况。通过这次实验，我学到了数据挖掘中聚类的重要性与基础性，可以实现多种聚类对简单数据集的操作。在实验过程中，也有许多不足，例如，实验规模小，无法进一步了解聚类算法更多的实用性价值。希望在以后实验中更加深入学习。
