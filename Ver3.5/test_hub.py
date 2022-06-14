import numpy as np
import faiss

if __name__ == '__main__':
    Kmeans_path        = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
    Kmeans_ID_path     = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'

    k = 4
    d = 128
    # SIFT1M = np.load("./input/SIFT1M/SIFT1M.npy")
    SIFT1M = np.load(Kmeans_path.format(str(0)))
    SIFT1M = np.float32(SIFT1M)
    index  = faiss.IndexFlatL2(d)
    index.add(SIFT1M)
    D, I   = index.search(SIFT1M, k)
    # D = D[:,1]
    # std = np.std(D)
    # mean = np.mean(D)
    # print(std)
    # print(mean)
    # print(D[:,1:])
    # D = D[:,1:]
    score = np.zeros(SIFT1M.shape[0],dtype=int)
    for fir in range (I.shape[0]):
        for sec in range (k):
            score[I[fir, sec]] += 1
    
    sort_ID = np.argsort(score)
    top_50 = SIFT1M[sort_ID[:int(SIFT1M.shape[0]*0.5)]]
    end_50 = SIFT1M[sort_ID[int(SIFT1M.shape[0]*0.5):]]
    top_50_ID = SIFT1M[sort_ID[:int(SIFT1M.shape[0]*0.5)]]
    end_50_ID = SIFT1M[sort_ID[int(SIFT1M.shape[0]*0.5):]]
    index.reset()
    index.add(end_50)
    D, I   = index.search(end_50, 2)
    D = D[:,1]
    std = np.std(D)
    mean = np.mean(D)
    print(max(D))
    print(std)
    print(mean)
    index.reset()
    index.add(top_50)
    D, I   = index.search(top_50, 2)
    D = D[:,1]
    std = np.std(D)
    mean = np.mean(D)
    print(std)
    print(mean)
    