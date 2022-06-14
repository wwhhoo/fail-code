import numpy as np
import faiss


data = np.load('./input/SIFT1M/SIFT1M.npy')
data = np.float32(data)
save_path    = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
save_ID_path = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
ncentroids = 256
niter      = 200
verbose    = True
d          = data.shape[1]
kmeans     = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(data)

kmeans.nprobe = niter
D, I = kmeans.index.search(data, 1)

count       = np.zeros(256,dtype=int)
output_data = np.zeros((256,1000000,128),dtype=np.uint8)
data_ID     = np.zeros((256,1000000),dtype=np.uint32)
data        = np.uint8(data)

for i in range(data.shape[0]):
   
    output_data[I[i],count[I[i]]] = data[i]
    data_ID[I[i],count[I[i]]]     = i
    count[I[i]] += 1

for i in range(output_data.shape[0]):
    np.save(save_path.format(str(i)), output_data[i,:count[i]])
    np.save(save_ID_path.format(str(i)), data_ID[i,:count[i]])
np.save('./input/kmeans_centroids.npy',kmeans.centroids)