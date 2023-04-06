from sklearn.datasets import make_blobs
Data, y=make_blobs(n_samples=600,n_features=2,centers=4)

import matplotlib.pyplot as plt
plt.scatter(Data[:,0], Data[:,1])
plt.show()

from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist 
import numpy as np 

# Méthode 1  de coude 

X = np.array(list(zip(Data[:,0], Data[:,1]))).reshape(len(Data[:,0]), 2) 
#Distorsion: elle est calculée comme la moyenne des distances
#au carré des centres de cluster des clusters respectifs. 
#En règle générale, la métrique de distance euclidienne est utilisée.

#Inertie: C’est la somme des distances au carré des échantillons par 
#rapport à leur centre de cluster le plus proche.
distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(2,10) 
  
for k in K: 
    
    kmeanModel = KMeans(n_clusters=k).fit(X) 
    kmeanModel.fit(X)     
      
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
    mapping2[k] = kmeanModel.inertia_ 
    
for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val)) 
    
plt.plot(K, distortions, 'bx-') 
plt.show() 

for key,val in mapping2.items(): 
    print(str(key)+' : '+str(val)) 

plt.plot(K, inertias, 'bx-') 
plt.show() 


# Méthode 2  de coude 

#Il s'agit essentiellement de la somme des carrés des distances entre 
#nos points de données et leur centre de gravité de cluster.




"""
for i in range(2,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(Data)
    Label = kmeans.labels_
    Centre = kmeans.cluster_centers_
    plt.scatter(
        Data[:, 0], Data[:, 1], c=Label, cmap ="plasma"
    )
    
    #plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],c='g',cmap="plasma")
    plt.title("KMeans clustering on sample data with n_clusters = %d " %i)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()


from sklearn.metrics import silhouette_samples, silhouette_score   
k = range(2,11)
for i in k:
     km = KMeans(n_clusters=i)
     km = km.fit(Data)
     silhouette_avg = silhouette_score(Data,km.labels_)
     print("For n_clusters =",i,"The average silhouette_score is :",silhouette_avg,)
  
from yellowbrick.cluster import SilhouetteVisualizer

k = range(2,11)
for n_clusters in k :
    plt.subplot(1,2,1)
    model = KMeans(n_clusters)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(Data) #Fit the data to the visualizer
    m = visualizer.silhouette_score_
    print("For n_clusters =",n_clusters,"The average silhouette_score is :",m,)
    plt.subplot(1,2,2)
    plt.scatter(Data[:,0],Data[:,1],c=model.labels_,cmap="plasma")
    visualizer.show()


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(Data, method='ward'))
plt.axhline(y=100, color='r', linestyle='--')


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
YPred=cluster.fit_predict(Data)
 
plt.figure(figsize=(10, 7))  
plt.scatter(Data[:,0], Data[:,1], c=YPred, cmap='plasma') 

# DBSCAN

from sklearn.cluster import DBSCAN
# cluster the data into five clusters
dbscan = DBSCAN(eps = 8, min_samples = 4).fit(Data) # fitting the model
labels = dbscan.labels_ # getting the labels

# Plot the clusters
plt.scatter(Data[:,0],Data[:,1], c = labels, cmap= "plasma") # plotting the clusters
plt.xlabel("Income") # X-axis label
plt.ylabel("Spending Score") # Y-axis label
plt.show() # showing the plot

"""









intertia = []

K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    intertia.append(km.inertia_)
    
plt.plot(K, intertia, marker= "x")
plt.xlabel('k')
plt.xticks(np.arange(10))
plt.ylabel('Intertia')
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
Label = kmeans.labels_
Centre = kmeans.cluster_centers_
plt.scatter(
    X[:, 0], X[:, 1], c=Label, cmap ="plasma"
)

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],c='g',cmap="plasma")
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()



# Le cofficient de Silhouette

from sklearn.metrics import silhouette_samples, silhouette_score
"""
# Calculate Silhoutte Score
score = silhouette_score(X, km.labels_, metric='euclidean')
# Print the score
print('Silhouetter Score: %.4f' % score)
"""

import matplotlib.cm as cm
range_n_clusters = [2, 3, 4]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()




range_n_clusters = [2, 3, 4]

for n_clusters in range_n_clusters:
    
    print("---------------------------------------")
    print("For n_clusters =",n_clusters)
    print("Silhouette score:",silhouette_score(X, km.labels_))
  
s = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(Data)
    label = kmeans.labels_
    s.append(silhouette_score(Data, label))
