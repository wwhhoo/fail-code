import numpy as np
import faiss                   # make faiss available
import math

import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

d = 128                         # dimension
knn = 16
dataset = np.load('./input/SIFT1M/SIFT1M.npy')#('./input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_0.npy')#
dataset = np.float32(dataset)
index = faiss.IndexFlatL2(d)
res = faiss.StandardGpuResources()
# Set GPU
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
# add vectors to the index
gpu_index_flat.add(dataset)
# search
D, I = gpu_index_flat.search(dataset, knn) # sanity check
score = np.zeros(dataset.shape[0],dtype=int)
for fir in range (I.shape[0]):
    for sec in range (knn):
        score[I[fir, sec]] += 1

sort_ID = np.argsort(score)
# print(dataset.shape[0])
top_50 = dataset[sort_ID[:int(dataset.shape[0]/2)]]
end_50 = dataset[sort_ID[int(dataset.shape[0]/2):]]
# print(end_50.shape)
top_50_embedded = TSNE(n_components=2).fit_transform(top_50)
end_50_embedded = TSNE(n_components=2).fit_transform(end_50)
# pca = PCA(n_components=2)
# top_reduced = pca.fit_transform(top_50)
# end_reduced = pca.fit_transform(end_50)

# We need a 2 x 944 array, not 944 by 2 (all X coordinates in one list)
# print(reduced.shape)
top = top_50_embedded.transpose()
end = end_50_embedded.transpose()
# print(t.shape)

plt.scatter(top[0], top[1],color='r', marker='x', label='hub')
plt.scatter(end[0], end[1],color='b', marker="o", label='anti-hub')
plt.legend()
plt.show()