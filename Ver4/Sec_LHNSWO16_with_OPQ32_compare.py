import numpy as np
import faiss
import time
from sklearn.linear_model import LinearRegression

# data path load: Kmeans save: HNSW

OPQ_data_path       = "./input/HNSWOPQ_with_OPQ_ver4/Kmeans_high_data/Sec_OPQ_data_{}.npy"
OPQ_ID_path         = "./input/HNSWOPQ_with_OPQ_ver4/Kmeans_high_ID/Sec_OPQ_{}_ID.npy"
OPQ_data_save_path  = "./input/HNSWOPQ_with_OPQ_ver4/OPQ32_data/Sec_OPQ_data_{}.npy"

d     = 128
maxN  = 8
index = 0
C     = 256

eFSearch = 100

if __name__ == '__main__':

    OPQ_train          = np.load("./input/SIFT1M/SIFT1M.npy")
    OPQindex_name      = "./input/HNSWOPQ_with_OPQ_ver4/OPQ_index/index_OPQ.index"

    bits  = 8
    nlist = 2**bits
    m     = 64

    OPQ_train = np.float32(OPQ_train)
    opq_index = faiss.ProductQuantizer(d, m, bits)
    OPQMatrix = faiss.OPQMatrix(d, m)
    OPQMatrix.pq = opq_index
    OPQMatrix.train(OPQ_train)
    OPQ_train = OPQMatrix.apply_py(OPQ_train)
    OPQquantizer = faiss.IndexFlatL2(d)
    OPQindex = faiss.IndexIVFPQ(OPQquantizer, d, nlist, m, bits)
    OPQindex.train(OPQ_train)

    query = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
    query = np.float32(query)
    OPQquery = OPQMatrix.apply_py(query)
    np.save("./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_Query.npy", OPQquery)

    np.save("./input/HNSWOPQ_with_OPQ_ver3.5/totalO_OPQ.npy", OPQ_train)
    faiss.write_index(OPQindex, OPQindex_name)

    for fir in range (C):

        # OPQ high
        OPQ_data = np.load(OPQ_data_path.format(str(fir)))
        OPQ_data_ID = np.load(OPQ_ID_path.format(str(fir)))
        OPQ_data = np.float32(OPQ_data)
        OPQ_data = OPQMatrix.apply_py(OPQ_data)

        np.save(OPQ_data_save_path.format(str(fir)), OPQ_data)
