from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import sys
import logging
from optparse import OptionParser
import warnings
from time import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
print("Loading 20 newsgroups dataset for categories:")
print(categories)

# 采集数据
# subset就是train,test,all三种可选，分别对应训练集、测试集和所有样本。
# categories:是指类别，如果指定类别，就会只提取出目标类，如果是默认，则是提取所有类别出来。
# shuffle:是否打乱样本顺序，如果是相互独立的话。
# random_state:打乱顺序的随机种子
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

# 特征提取
print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X.toarray(), quantile=0.2, n_samples=500)

# #############################################################################
# Do the actual clustering

print(68 * '_')
print('name\t\ttime\t\thomo\t\tcompl\t\tNMI')


# 对该次实验进行评价,分别用到几种评价指标同质性分数，完整性分数，以及两者的调和平均，还有规范互信息分数
# 同质性表示每个划分的簇只包含来自同一类别
# 完整性表示来自同一类别的样本是否被划分到同一个簇
def bench(estimator, name, data):
    t0 = time()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. completing it to avoid stopping the tree early.",
            category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding may not work as expected.", category=UserWarning)

    estimator.fit(data)
    print('%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.normalized_mutual_info_score(labels, estimator.labels_)
             ))


bench(KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=opts.verbose), name="KM", data=X)
bench(SpectralClustering(n_clusters=true_k, gamma=0.1, eigen_solver='arpack'), name="SC", data=X)
bench(DBSCAN(eps=0.5, min_samples=1), name="DB", data=X)
bench(AffinityPropagation(), name="AP", data=X)
bench(MeanShift(bandwidth=bandwidth, bin_seeding=True), name="MS", data=X.toarray())
bench(AgglomerativeClustering(n_clusters=true_k, linkage='ward'), name="ward", data=X.toarray())
bench(AgglomerativeClustering(n_clusters=true_k, linkage='complete'), name="AC", data=X.toarray())

gm = GaussianMixture(n_components=true_k)
t0 = time()
gm.fit(X.toarray())
print('%s\t\t%.4f\t%.4f\t\t%.4f\t\t%.4f' % ('GM', time() - t0, metrics.homogeneity_score(labels, gm.predict(X.toarray())), metrics.completeness_score(labels, gm.predict(X.toarray())),metrics.normalized_mutual_info_score(labels, gm.predict(X.toarray()))))

print(68 * '_')