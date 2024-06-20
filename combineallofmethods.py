import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from FRUFS import FRUFS
from lightgbm import LGBMClassifier, LGBMRegressor
from utils import plot_confusion_matrix,silhouette_score,f_a_score,to_csv,get_distance,fs_FRUFS,getMojofm,get_similarity,plot_count,getFeatureMatch
import matplotlib.pyplot as plt
import numpy as np
from ExKMC.Tree import Tree
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from ShallowTree.ShallowTree import ShallowTree
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



DF = pd.read_csv('data/family.csv',index_col=False)
print(f'Data shape is {DF.shape}')

grouped = DF.groupby(['family'])
# get numbers of each group
groupCount = grouped['family'].count()
table_count = groupCount.sort_values(ascending=False)[:20]
print(table_count)
# select by numbers of the top
# TOP = 11
TOP = 10

selected = groupCount.sort_values(ascending=False)[:TOP]
# plot_count(selected)

# get names
selectedNames = list(selected.index)
print(f"selected family are {selectedNames}")

# select trained
train = DF.loc[DF["family"].isin(selectedNames)]
train.reset_index(inplace=True, drop=True)
print(f"train dataset shape {train.shape}")


# encode into encoder
le = LabelEncoder()
label = le.fit_transform(train["family"])
# print(train.head(5))

# exclue name -> X_
train = train.loc[:, train.columns != "name"]
train = train.loc[:, train.columns != "family"]

X = train

# with X.index canbe reserved to original labels
Y = np.take(label, X.index)

X.reset_index(inplace=True, drop=True)

#** K is here  ******

K = len(selectedNames)

X_train = X
print(X_train.shape)

########### sample balance ######
clf = NearMiss(version=1)
print('before sampling')
print(sorted(Counter(Y).items()))
resampled_X, resampled_y = clf.fit_resample(X_train, Y)
print('after sampling')
print(sorted(Counter(resampled_y).items()))
X_train_ = pd.DataFrame(resampled_X)

########## feature selection #####

# feature selection
# k = float(0.1155555555555555555555)
# 0.035-67.83  0.08- 67.43
# X_train_prued = fs_FRUFS(X_train_,k=0.3,display=True,iter=0)
    # 0.38 best
# k = 0.12 #for sclaler is the best

k = 0.38
fea_model = FRUFS(model_c=LGBMClassifier(random_state=25), k=k, n_jobs=-1)
X_train_prued = fea_model.fit_transform(X_train_)

################# start here to compare different explanation methods ######

#***** 1st is shallowtree ********

tree = ShallowTree(K)

kmeans = KMeans(K,random_state=0)
kmeans.fit(X_train_prued)
tree.fit(X_train_prued.values)
#prediction = tree.fit_predict(X_train_prued.values)
tree.plot('shallowtree',feature_names= X_train_prued.columns.values)

Y_sampled = pd.DataFrame(resampled_y,columns=['cluster'])
print(Y_sampled['cluster'])


##### 2ed is mpc

from mpc import *
X, labels, w, b = MPCPolytopeOpt(X_train_prued.values, 10, metric='silhouette', card=1, M=1, verbose=True)
print(w,b)
print('labels',labels)
def interpret_rules_for_cluster(cluster_label, X, labels, w, b, feature_names):
    # Identify data points in the cluster
    cluster_points = X[labels == cluster_label]

    # Initialize rules for the cluster
    cluster_rules = set()
    important_features = set()

    # Check each hyperplane
    for edge, weights in w.items():
        consistent = True
        for point in cluster_points:
            value = np.dot(weights, point) + b[edge]
            if value > 0:
                consistent = False
                break

        # If all points in the cluster satisfy the hyperplane, add its rule
        if consistent:
            rule = []
            for i, weight in enumerate(weights):
                if weight != 0:
                    important_features.add(feature_names[i])
                    if weight > 0:
                        rule.append(f'{feature_names[i]} >= {-b[edge]/weight:.2f}')
                    else:
                        rule.append(f'{feature_names[i]} <= {-b[edge]/weight:.2f}')
            cluster_rules.add(' AND '.join(rule))

    return list(cluster_rules), list(important_features)


feature_names = X_train_prued.columns.values

# Interpret each cluster
for cluster_label in np.unique(labels):
    rules, features = interpret_rules_for_cluster(cluster_label, X, labels, w, b, feature_names)
    print(f'Cluster {cluster_label}:')
    for rule in rules:
        print(f'- {rule}')
    print(f'Important Features: {", ".join(features)}')
    print()


#******* 3rd is exkmc ******


tree = Tree(k=K, max_leaves=2*K)

# Construct the tree, and return cluster labels
prediction = tree.fit_predict(X_train_prued)

# Tree plot saved to filename
tree.plot(filename='exkmc11',feature_names= X_train_prued.columns.values)



##### 4th is cart

# Assuming you have your feature matrix 'X' and target variable 'y' ready

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_train_prued, km.labels_, test_size=0.2, random_state=42)

# Create an instance of the DecisionTreeClassifier
kmeans = KMeans(K,random_state=0)
kmeans.fit(X_train_prued)
clf = DecisionTreeClassifier(max_leaf_nodes = 6*10)

# Fit the classifier to the training data
clf.fit(X_train_prued, kmeans.labels_)

# Predict the labels for the test set
y_pred = clf.predict(X_train_prued)

#showcarttree
features = list(X_train_prued.columns.values)
plt.figure(figsize=(250,50))
plot_tree(clf, filled=True,fontsize=15,feature_names=features)
plt.savefig('carttree_small_leaves.png')





