import numpy as np
import faiss
import time
from sklearn.linear_model import LinearRegression

# data path load: Kmeans save: HNSW
HNSW_data_path      = './input/HNSWOPQ_with_OPQ_ver4/Kmeans_low_data/Sec_HNSW_{}.npy'
HNSW_ID_path        = './input/HNSWOPQ_with_OPQ_ver4/Kmeans_low_ID/Sec_HNSW_{}_ID.npy'
HNSW_data_save_path = "./input/HNSWOPQ_with_OPQ_ver4/OPQ16_data/Sec_HNSW_{}.npy"
OPQ_data_path       = "./input/HNSWOPQ_with_OPQ_ver4/Kmeans_high_data/Sec_OPQ_data_{}.npy"
OPQ_ID_path         = "./input/HNSWOPQ_with_OPQ_ver4/Kmeans_high_ID/Sec_OPQ_{}_ID.npy"
OPQ_data_save_path  = "./input/HNSWOPQ_with_OPQ_ver4/OPQ32_data/Sec_OPQ_data_{}.npy"
concate_data_path   = "./input/HNSWOPQ_with_OPQ_ver4/OPQ_data/Sec_OPQ_data_{}.npy"
concate_ID_path     = "./input/HNSWOPQ_with_OPQ_ver4/OPQ_ID/Sec_OPQ_{}_ID.npy"

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

    OPQ_train          = np.load("./input/HNSWOPQ_with_OPQ_ver4/total_OPQ.npy")
    HNSW_OPQ_train     = np.load("./input/HNSWOPQ_with_OPQ_ver4/total_HNSW.npy")
    OPQindex_name      = "./input/HNSWOPQ_with_OPQ_ver4/OPQ_index/index_OPQ.index"
    HNSW_OPQindex_name = "./input/HNSWOPQ_with_OPQ_ver4/OPQ_index/HNSW_OPQindex.index"

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
    np.save("./input/HNSWOPQ_with_OPQ_ver4/OPQ_Query.npy", OPQquery)
    faiss.write_index(OPQindex, OPQindex_name)

    m = 32
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
    np.save("./input/HNSWOPQ_with_OPQ_ver4/HNSW_Query.npy", HNSWquery)
    
    FlatIndex = faiss.IndexFlatL2(d)
    knn = 9
    OPQ_loss = 0.0
    zero_pad_loss = 0.0

    for fir in range (C):

        # OPQ high
        OPQ_data = np.load(OPQ_data_path.format(str(fir)))
        OPQ_data_ID = np.load(OPQ_ID_path.format(str(fir)))
        OPQ_data = np.float32(OPQ_data)
        OPQ_data = OPQMatrix.apply_py(OPQ_data)

        np.save(OPQ_data_save_path.format(str(fir)), OPQ_data)
        
        # OPQ low
        HNSW_data    = np.load(HNSW_data_path.format(str(fir)))
        HNSW_data_ID = np.load(HNSW_ID_path.format(str(fir)))
        HNSW_data    = np.float32(HNSW_data)
        HNSW_data    = HNSW_OPQMatrix.apply_py(HNSW_data)

        # numbers      = OPQ_data.shape[0]
        # compute_keys = True
        # # Keys
        # list_nos = np.zeros(numbers)
        # list_nos = np.int64(list_nos)

        # codes = np.empty((numbers,OPQindex.code_size), dtype=np.uint8)
        # # encoder
        # OPQindex.encode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(OPQ_data), faiss.swig_ptr(codes),compute_keys )

        # OPQindex.nprobe = nlist              # make comparable with experiment above
        # xcodes = np.empty((numbers,d), dtype=np.float32)
        # # decoder
        # OPQindex.decode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes)) 
        # # print(xcodes.shape)
        # OPQ_loss += np.sum((xcodes - OPQ_data)**2)


        numbers      = HNSW_data.shape[0]
        compute_keys = True
        # Keys
        list_nos = np.zeros(numbers)
        list_nos = np.int64(list_nos)

        codes = np.empty((numbers,HNSW_OPQindex.code_size), dtype=np.uint8)
        # encoder
        HNSW_OPQindex.encode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(HNSW_data), faiss.swig_ptr(codes),compute_keys )

        HNSW_OPQindex.nprobe = nlist              # make comparable with experiment above
        xcodes = np.empty((numbers,d), dtype=np.float32)
        # decoder
        HNSW_OPQindex.decode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes)) 
        OPQ_loss += np.sum((xcodes - HNSW_data)**2)

        
        FlatIndex.add(HNSW_data)
        D, I = FlatIndex.search(HNSW_data, knn)
        FlatIndex.reset()
        topk_data = xcodes[I]
        # print(topk_data.shape)
        quanti_data = np.zeros((xcodes.shape[0]*xcodes.shape[1],knn))
        # print(quanti_data.shape)
        # -------------------------------------------------------------------------
        # for i in range(xcodes.shape[0]):
        #     tran_topk_data = np.transpose(topk_data[i])
        #     quanti_data[i*d:(i+1)*d] = tran_topk_data
        # tran_quanti_data = np.transpose(quanti_data)
        # reshape_HNSW_data = np.reshape(HNSW_data, (int(HNSW_data.shape[0]*HNSW_data.shape[1]),1))
        # Beta = tran_quanti_data.dot(reshape_HNSW_data)
        # # new_HNSW_data = np.transpose(Beta).dot(tran_quanti_data)
        # new_HNSW_data = quanti_data.dot(Beta)
        # new_HNSW_data = np.reshape(new_HNSW_data, (int(new_HNSW_data.shape[0]/d),d))

        # -------------------------------------------------------------------------
        # topk_data = xcodes[I]
        # # print(topk_data.shape)
        # quanti_data = np.zeros((xcodes.shape[0]*xcodes.shape[1],knn))
        # # print(quanti_data.shape)
        # see_weight = np.zeros((1,knn))
        # see_weight[0,0] = 1
        # for i in range(xcodes.shape[0]):
        #     tran_topk_data = np.transpose(topk_data[i])
        #     quanti_data[i*d:(i+1)*d] = tran_topk_data
        # reshape_HNSW_data = np.reshape(HNSW_data, (int(HNSW_data.shape[0]*HNSW_data.shape[1]),1))
        # # print(reshape_HNSW_data.shape)
        # tran_quanti_data = np.transpose(quanti_data)
        # reg = LinearRegression().fit(tran_quanti_data.dot(quanti_data), tran_quanti_data.dot(reshape_HNSW_data))
        # # reg = LinearRegression().fit(quanti_data, reshape_HNSW_data)
        # # print(reg.score(quanti_data, reshape_HNSW_data))
        # new_HNSW_data = reg.predict(quanti_data)
        # # see_weight = reg.predict(see_weight)
        # # print(see_weight)
        
        # new_HNSW_data = np.reshape(new_HNSW_data, (int(new_HNSW_data.shape[0]/d),d))
        # -------------------------------------------------------------------------
        # topk_data = xcodes[I]
        quanti_data = np.zeros((128,knn))
        test_HNSW_data = np.zeros((HNSW_data.shape))
        for i in range(HNSW_data.shape[0]):
            # for i in range(xcodes.shape[0]):
            tran_topk_data = np.transpose(topk_data[i])
            quanti_data = tran_topk_data
            reshape_HNSW_data = np.reshape(HNSW_data[i], (128,1))
            reg = LinearRegression().fit(quanti_data, reshape_HNSW_data)
            new_HNSW_data = reg.predict(quanti_data)
            # print(reg.score(quanti_data, reshape_HNSW_data))
            new_HNSW_data = np.reshape(new_HNSW_data, (1,d))
            test_HNSW_data[i] = new_HNSW_data


        zero_pad_loss += np.sum((new_HNSW_data - HNSW_data)**2)
        



        np.save(HNSW_data_save_path.format(str(fir)), new_HNSW_data)
        # save as one data
        np.save(concate_data_path.format(str(fir)),np.concatenate((OPQ_data,new_HNSW_data)))
        np.save(concate_ID_path.format(str(fir)),np.concatenate((OPQ_data_ID,HNSW_data_ID)))

    print(OPQ_loss)
    print(zero_pad_loss)