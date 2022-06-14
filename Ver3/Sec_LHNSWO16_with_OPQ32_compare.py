import numpy as np
import faiss
import time

# data path load: Kmeans save: HNSW
HNSW_index_save     = "./input/HNSWOPQ_with_OPQ_ver3/HNSW_index/index_HNSW_{}.index"
HNSW_data_save_path = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_data_with_centorid/Sec_HNSW_{}.npy"
HNSW_ID_save_path   = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID_with_centorid/Sec_HNSW_data_ID_{}.npy"
OPQ_data_path       = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256/Sec_Kmeans{}_{}.npy"
OPQ_ID_path         = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID/Sec_Kmeans{}_{}_ID.npy"
OPQ_ID_save_path    = "./input/HNSWOPQ_with_OPQ_ver3/Kmeans256_ID/Sec_Kmeans256_{}_ID.npy"
OPQ_save_path       = "./input/HNSWOPQ_with_OPQ_ver3/OPQ32_data/Sec_OPQ_data_256_{}.npy"
centorid_path       = "./input/HNSWOPQ_with_OPQ_ver3/centroids/kmeans_centroids_{}.npy"

d     = 128
maxN  = 8
efCon = 200
index = 0
C     = 256
SC    = 256
eFSearch = 100

if __name__ == '__main__':

    OPQ_train          = np.load("./input/SIFT1M/SIFT1M.npy")
    OPQindex_name      = "./input/HNSWOPQ_with_OPQ_ver3/OPQ_index/index_OPQ.index"

    bits  = 8
    nlist = 2**bits
    m     = 32

    OPQ_train      = np.float32(OPQ_train)

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
    np.save("./input/HNSWOPQ_with_OPQ_ver3/OPQ_Query.npy", OPQquery)
    faiss.write_index(OPQindex, OPQindex_name)

    Sec_clu_num = np.zeros((C,SC),dtype=int)

    for fir in range (C):
        conbine_sec = np.zeros((100000,d))
        conbine_ID  = np.zeros((100000))
        sec_index = 0
        for sec in range (SC):
            # load data
            Kmeans_data = np.load(OPQ_data_path.format(str(fir), str(sec)))
            Kmeans_data_ID = np.load(OPQ_ID_path.format(str(fir), str(sec)))
            Kmeans_data = np.float32(Kmeans_data)
            # compute_keys = True
            Kmeans_data = OPQMatrix.apply_py(Kmeans_data)

            num = Kmeans_data.shape[0]
            conbine_sec[sec_index:sec_index+num] = Kmeans_data
            conbine_ID[sec_index:sec_index+num] = Kmeans_data_ID
            sec_index += num
            Sec_clu_num[fir,sec] = num
            

        np.save(OPQ_save_path.format(str(fir)), conbine_sec[:sec_index])
        np.save(OPQ_ID_save_path.format(str(fir)), conbine_ID[:sec_index])

        # put centorid into HNSW
        centorid          = np.load(centorid_path.format(str(fir)))

        kmeans_cent = SC
        if centorid.shape[0] < SC:
            kmeans_cent = centorid.shape[0]

        new_HNSW_data    = centorid
        new_HNSW_data_ID = np.zeros((SC))
        for sec in range (kmeans_cent):
            new_HNSW_data_ID[sec] = fir*1000 + sec

        new_HNSW_data = np.float32(new_HNSW_data)

        HNSWindex = faiss.IndexHNSWFlat(d, maxN)
        HNSWindex.hnsw.efConstruction = efCon
        HNSWindex.hnsw.efSearch = eFSearch
        HNSWindex.verbose = True
        HNSWindex.add(new_HNSW_data)


        faiss.write_index(HNSWindex, HNSW_index_save.format(str(fir)))

        np.save(HNSW_data_save_path.format(str(fir)), new_HNSW_data)
        np.save(HNSW_ID_save_path.format(str(fir)), new_HNSW_data_ID)
    
    np.save("./input/HNSWOPQ_with_OPQ_ver3/Second_cluster_data_num.npy",Sec_clu_num)

