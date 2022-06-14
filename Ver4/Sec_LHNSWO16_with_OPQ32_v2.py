import numpy as np
import faiss
import time

# data path load: Kmeans save: HNSW
HNSW_data_path      = './input/HNSWOPQ_with_OPQ_ver3.5/Kmeans_low_data/Sec_HNSW_{}.npy'
HNSW_ID_path        = './input/HNSWOPQ_with_OPQ_ver3.5/Kmeans_low_ID/Sec_HNSW_{}_ID.npy'
HNSW_data_save_path = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ16_data/Sec_HNSW_{}.npy"
OPQ_data_path       = "./input/HNSWOPQ_with_OPQ_ver3.5/Kmeans_high_data/Sec_OPQ_data_{}.npy"
OPQ_ID_path         = "./input/HNSWOPQ_with_OPQ_ver3.5/Kmeans_high_ID/Sec_OPQ_{}_ID.npy"
OPQ_data_save_path  = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ32_data/Sec_OPQ_data_{}.npy"
concate_data_path   = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_data/Sec_OPQ_data_{}.npy"
concate_ID_path     = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_ID/Sec_OPQ_{}_ID.npy"

d     = 128
maxN  = 8
efCon = 200
index = 0
C     = 256
SC    = 256
HSC   = 128
LSC   = 128
eFSearch = 100

if __name__ == '__main__':

    OPQ_train          = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/total_OPQ.npy")
    HNSW_OPQ_train     = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/total_HNSW.npy")
    OPQindex_name      = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_index/index_OPQ.index"
    HNSW_OPQindex_name = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_index/HNSW_OPQindex.index"

    bits  = 8
    nlist = 2**bits
    m     = 64

    OPQ_train      = np.float32(OPQ_train)
    HNSW_OPQ_train = np.float32(HNSW_OPQ_train)

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
    faiss.write_index(OPQindex, OPQindex_name)

    m = 64
    # cent_index = 0
    HNSW_opq_index = faiss.ProductQuantizer(d, m, bits)
    HNSW_OPQMatrix = faiss.OPQMatrix(d, m)
    HNSW_OPQMatrix.pq = HNSW_opq_index
    HNSW_OPQMatrix.train(HNSW_OPQ_train)
    train_with_cent = HNSW_OPQMatrix.apply_py(HNSW_OPQ_train)
    HNSW_OPQquantizer = faiss.IndexFlatL2(d)
    HNSW_OPQindex = faiss.IndexIVFPQ(HNSW_OPQquantizer, d, nlist, m, bits)
    HNSW_OPQindex.train(train_with_cent)

    faiss.write_index(HNSW_OPQindex, HNSW_OPQindex_name)

    HNSWquery = HNSW_OPQMatrix.apply_py(query)
    np.save("./input/HNSWOPQ_with_OPQ_ver3.5/HNSW_Query.npy", HNSWquery)

    for fir in range (C):

        # load data
        OPQ_data = np.load(OPQ_data_path.format(str(fir)))
        OPQ_data_ID = np.load(OPQ_ID_path.format(str(fir)))
        OPQ_data = np.float32(OPQ_data)
        OPQ_data = OPQMatrix.apply_py(OPQ_data)

        np.save(OPQ_data_save_path.format(str(fir)), OPQ_data)

        HNSW_data    = np.load(HNSW_data_path.format(str(fir)))
        HNSW_data_ID = np.load(HNSW_ID_path.format(str(fir)))
        HNSW_data  = np.float32(HNSW_data)
        HNSW_data = HNSW_OPQMatrix.apply_py(HNSW_data)

        np.save(HNSW_data_save_path.format(str(fir)), HNSW_data)

        np.save(concate_data_path.format(str(fir)),np.concatenate((OPQ_data,HNSW_data)))
        np.save(concate_ID_path.format(str(fir)),np.concatenate((OPQ_data_ID,HNSW_data_ID)))
