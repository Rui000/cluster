import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from FRUFS import FRUFS
from lightgbm import LGBMClassifier
from sklearn.metrics import silhouette_score
from ShallowTree.ShallowTree import ShallowTree
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from ExKMC.Tree import Tree
from mpc import *

from sklearn.tree import DecisionTreeClassifier,plot_tree
# Function to calculate Dunn Index
def dunn_index(X, labels):
    distances = squareform(pdist(X))
    unique_labels = np.unique(labels)
    min_intercluster = np.inf
    max_intracluster = 0

    for i in unique_labels:
        for j in unique_labels:
            if i != j:
                intercluster_distance = np.min(distances[labels == i][:, labels == j])
                min_intercluster = min(min_intercluster, intercluster_distance)

        intracluster_distance = np.max(pdist(X[labels == i]))
        max_intracluster = max(max_intracluster, intracluster_distance)

    return min_intercluster / max_intracluster

# Load data
DF = pd.read_csv('data/family.csv', index_col=False)
print(f'Data shape is {DF.shape}')

grouped = DF.groupby(['family'])
groupCount = grouped['family'].count()
selected = groupCount.sort_values(ascending=False)[:10]
selectedNames = list(selected.index)
train = DF.loc[DF["family"].isin(selectedNames)]
train.reset_index(inplace=True, drop=True)

le = LabelEncoder()
label = le.fit_transform(train["family"])
train = train.loc[:, train.columns != "name"]
train = train.loc[:, train.columns != "family"]
X = train
Y = np.take(label, X.index)
X.reset_index(inplace=True, drop=True)
K = len(selectedNames)

# Sampling balance
clf = NearMiss(version=1)
resampled_X, resampled_y = clf.fit_resample(X, Y)
X_train_ = pd.DataFrame(resampled_X)

# Feature selection
k = 0.38
fea_model = FRUFS(model_c=LGBMClassifier(random_state=25), k=k, n_jobs=-1)
X_train_prued = fea_model.fit_transform(X_train_)
# Define the number of repetitions
num_repetitions = 100
# Initialize dictionaries to store the results
#silhouette_results = {'shallow_tree': [], 'exkmc': [], 'km': [], 'cart': [], 'mpc':[]}
#dunn_results = {'shallow_tree': [], 'exkmc': [], 'km': [], 'cart': [],'mpc':[]}

silhouette_results = {'shallow_tree': [], 'exkmc': [], 'km': [], 'cart': []}
dunn_results = {'shallow_tree': [], 'exkmc': [], 'km': [], 'cart': []}

for _ in range(num_repetitions):

    # ShallowTree
    tree = ShallowTree(K)
    prediction = tree.fit_predict(X_train_prued.values)
    silhouette_avg = round(silhouette_score(X_train_prued, prediction), 2)
    dunn_index_value = round(dunn_index(X_train_prued, prediction), 2)
    silhouette_results['shallow_tree'].append(silhouette_avg)
    dunn_results['shallow_tree'].append(dunn_index_value)

    # ExKMC
    tree = Tree(k=K, max_leaves=2*K)
    predictionex = tree.fit_predict(X_train_prued)
    silhouette_results['exkmc'].append(round(silhouette_score(X_train_prued, predictionex),2))
    dunn_results['exkmc'].append(round(dunn_index(X_train_prued, predictionex),2))

    # KMeans
    kmeans = KMeans(K, random_state=0)
    km = kmeans.fit(X_train_prued)
    treekm = Tree(k=K, max_leaves=2*K)
    treekm.fit(km.cluster_centers_)
    kmpred = treekm.predict(X_train_prued)
    silhouette_results['km'].append(silhouette_score(X_train_prued, kmpred))
    dunn_results['km'].append(dunn_index(X_train_prued, kmpred))

    # CART
    clf = DecisionTreeClassifier(max_leaf_nodes=6*K)
    clf.fit(X_train_prued, km.labels_)
    y_pred = clf.predict(X_train_prued)
    silhouette_results['cart'].append(round(silhouette_score(X_train_prued, y_pred),2))
    dunn_results['cart'].append(round(dunn_index(X_train_prued, y_pred),2))

    #mpc
    #X, labels, w, b = MPCPolytopeOpt(X_train_prued.values, 10, metric='silhouette', card=1, M=1, verbose=False)
    #silhouette_results['mpc'].append(silhouette_score(X_train_prued, labels))
    #dunn_results['mpc'].append(dunn_index(X_train_prued, labels))


# Compute averages
average_silhouette = {method: np.mean(values) for method, values in silhouette_results.items()}
average_dunn = {method: np.mean(values) for method, values in dunn_results.items()}

print("Average Silhouette Scores:", average_silhouette)
print("Average Dunn Indexes:", average_dunn)
print("Silhouette Scores",silhouette_results)
print("dunn_results",dunn_results)