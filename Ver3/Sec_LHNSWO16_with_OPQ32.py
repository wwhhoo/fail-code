import numpy as np
import faiss
import time

# data path load: Kmeans save: HNSW
HNSW_index_save     = "./input/HNSWOPQ_with_OPQ_ver3/HNSW_index/index_HNSW_{}.index"
HNSW_data_path      = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Level_HNSW_data/Sec_HNSW_{}.npy"
HNSW_data_save_path = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_data_with_centorid/Sec_HNSW_{}.npy"
HNSW_ID_path        = './input/HNSWOPQ_with_OPQ_ver3/Sec_Level_HNSW_dataID/Sec_HNSW_{}_ID.npy'
HNSW_ID_save_path   = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID_with_centorid/Sec_HNSW_data_ID_{}.npy"
OPQ_data_path       = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256/Sec_Kmeans{}_{}.npy"
OPQ_ID_path         = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID/Sec_Kmeans{}_{}_ID.npy"
# OPQ_ID_save_path    = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID/Sec_Kmeans{}_{}_ID.npy"
OPQ_ID_save_path    = "./input/HNSWOPQ_with_OPQ_ver3/Kmeans256_ID/Sec_Kmeans256_{}_ID.npy"
# OPQ_save_path         = "./input/HNSWOPQ_with_OPQ_ver3/OPQ32_data/Sec_OPQ_data_{}_{}.npy"
OPQ_save_path       = "./input/HNSWOPQ_with_OPQ_ver3/OPQ32_data/Sec_OPQ_data_256_{}.npy"
centorid_path       = "./input/HNSWOPQ_with_OPQ_ver3/centroids/kmeans_centroids_{}.npy"

d     = 128
maxN  = 8
efCon = 200
# PQNN  = 512
index = 0
C     = 256
SC    = 256
eFSearch = 100

if __name__ == '__main__':

    OPQ_train          = np.load("./input/HNSWOPQ_with_OPQ_ver3/total_OPQ.npy")
    HNSW_OPQ_train     = np.load("./input/HNSWOPQ_with_OPQ_ver3/total_HNSW.npy")
    OPQindex_name      = "./input/HNSWOPQ_with_OPQ_ver3/OPQ_index/index_OPQ.index"
    HNSW_OPQindex_name = "./input/HNSWOPQ_with_OPQ_ver3/OPQ_index/HNSW_OPQindex.index"

    bits  = 8
    nlist = 2**bits
    m     = 32

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
    np.save("./input/HNSWOPQ_with_OPQ_ver3/OPQ_Query.npy", OPQquery)
    faiss.write_index(OPQindex, OPQindex_name)

    m = 16
    total_centorid = np.zeros((SC*C,d))
    cent_index = 0
    for fir in range (C):
        centorid = np.load(centorid_path.format(str(fir)))
        total_centorid[cent_index:cent_index+centorid.shape[0]] = centorid
        cent_index += centorid.shape[0]
    train_with_cent = np.concatenate((HNSW_OPQ_train,total_centorid[:cent_index]))
    train_with_cent = np.float32(train_with_cent)
    HNSW_opq_index = faiss.ProductQuantizer(d, m, bits)
    HNSW_OPQMatrix = faiss.OPQMatrix(d, m)
    HNSW_OPQMatrix.pq = HNSW_opq_index
    HNSW_OPQMatrix.train(train_with_cent)
    train_with_cent = HNSW_OPQMatrix.apply_py(train_with_cent)
    HNSW_OPQquantizer = faiss.IndexFlatL2(d)
    HNSW_OPQindex = faiss.IndexIVFPQ(HNSW_OPQquantizer, d, nlist, m, bits)
    HNSW_OPQindex.train(train_with_cent)

    faiss.write_index(HNSW_OPQindex, HNSW_OPQindex_name)

    HNSWquery = HNSW_OPQMatrix.apply_py(query)
    np.save("./input/HNSWOPQ_with_OPQ_ver3/HNSW_Query.npy", HNSWquery)


    Sec_clu_num = np.zeros((C,SC),dtype=int)

    for fir in range (C):
        conbine_sec = np.zeros((10000,d))
        conbine_ID  = np.zeros((10000))
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
        HNSW_data         = np.load(HNSW_data_path.format(str(fir)))
        HNSW_data_ID      = np.load(HNSW_ID_path.format(str(fir)))
        new_HNSW_data     = np.zeros((HNSW_data.shape[0]+SC,HNSW_data.shape[1]))
        new_HNSW_data_ID  = np.zeros((HNSW_data_ID.shape[0]+SC))
        new_HNSW_data[:HNSW_data.shape[0]]       = HNSW_data
        new_HNSW_data_ID[:HNSW_data_ID.shape[0]] = HNSW_data_ID

        kmeans_cent = SC
        if centorid.shape[0] < SC:
            kmeans_cent = centorid.shape[0]

        for sec in range (kmeans_cent):
                
            new_HNSW_data[HNSW_data.shape[0]+sec]       = centorid[sec]
            new_HNSW_data_ID[HNSW_data_ID.shape[0]+sec] = (fir*1000 + sec+ 1000000)

        new_HNSW_data = np.float32(new_HNSW_data)
        new_HNSW_data = HNSW_OPQMatrix.apply_py(new_HNSW_data)
        num = new_HNSW_data.shape[0]
        compute_keys  = True
        list_nos = np.zeros(num)
        list_nos = np.int64(list_nos)
        codes = np.empty((num,HNSW_OPQindex.code_size), dtype=np.uint8)
        HNSW_OPQindex.encode_multiple(num, faiss.swig_ptr(list_nos), faiss.swig_ptr(new_HNSW_data), faiss.swig_ptr(codes),compute_keys )
        HNSW_OPQindex.nprobe = nlist
        xcodes = np.empty((num,d), dtype=np.float32)
        HNSW_OPQindex.decode_multiple(num, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes)) 


        HNSWindex = faiss.IndexHNSWFlat(d, maxN)
        HNSWindex.hnsw.efConstruction = efCon
        HNSWindex.hnsw.efSearch = eFSearch
        HNSWindex.verbose = True
        HNSWindex.add(xcodes)
        
        faiss.write_index(HNSWindex, HNSW_index_save.format(str(fir)))

        np.save(HNSW_data_save_path.format(str(fir)), new_HNSW_data)
        np.save(HNSW_ID_save_path.format(str(fir)), new_HNSW_data_ID)
    
    np.save("./input/HNSWOPQ_with_OPQ_ver3/Second_cluster_data_num.npy",Sec_clu_num)

