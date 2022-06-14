import numpy as np
import faiss
import time

# data path load: Kmeans save: HNSW
data_path    = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
ID_path      = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
index_name   = "./input/HNSW_with_PQ/HNSW_index/index_HNSW_{}.index"
HNSW_save    = "./input/HNSW_with_PQ/HNSW_data/HNSW_data_{}.npy"
HNSW_ID_save = "./input/HNSW_with_PQ/HNSW_data_ID/HNSW_data_ID_{}.npy"
PQ_save      = "./input/HNSW_with_PQ/PQ_data/PQ_data_{}.npy"
PQ_ID_save   = "./input/HNSW_with_PQ/PQ_data_ID/PQ_data_ID_{}.npy"
OPQ_save     = "./input/HNSWOPQ_with_OPQ/OPQ_data/OPQ_data_{}.npy"
centorid     = np.load('./input/kmeans_centroids.npy')

d     = 128
maxN  = 8
efCon = 200
PQNN  = 512
index = 0
C     = 256
eFSearch = 100


PQ_train    = np.zeros((C*PQNN,d))
PQ_train_ID = np.zeros((C*PQNN,1))

# for each kmeans clusters
min_data = 0
for i in range (C):
    # load data
    Kmeans_data    = np.load(data_path.format(str(i)))
    Kmeans_data    = np.float32(Kmeans_data)
    Kmeans_data_ID = np.load(ID_path.format(str(i)))
    # for HNSW
    HNSW_data      = np.zeros((Kmeans_data.shape[0]-PQNN+1,d))
    HNSW_data_ID   = np.zeros((Kmeans_data.shape[0]-PQNN+1))
    # find nearest PQNN data for PQ, other for HNSW
    flat_index      = faiss.IndexFlatL2(d)
    flat_index.add(Kmeans_data)
    single_centroid = centorid[i].reshape(1,centorid[i].shape[0])
    D, I = flat_index.search(single_centroid, PQNN)
    I    = np.sort(I)
    HNSW_index  = 0
    array_index = 0
    # devide data
    for j in range (PQNN):
        # start from faiss I
        PQ_train[i*PQNN+j]    = Kmeans_data[I[0,j]].reshape(1,Kmeans_data[I[0,j]].shape[0])
        PQ_train_ID[i*PQNN+j] = Kmeans_data_ID[I[0,j]]
        
        if array_index != I[0,j]:
            HNSW_data[HNSW_index:(HNSW_index+I[0,j]-array_index)]    = Kmeans_data[array_index:I[0,j]]
            HNSW_data_ID[HNSW_index:(HNSW_index+I[0,j]-array_index)] = Kmeans_data_ID[array_index:I[0,j]]
            HNSW_index  = (HNSW_index+I[0,j]-array_index)
            array_index = (I[0,j]+1)
        else:
            array_index += 1

    # np.save(PQ_save.format(str(i)), PQ_train[PQNN*i:PQNN*(i+1)])
    # np.save(PQ_ID_save.format(str(i)), PQ_train_ID[PQNN*i:PQNN*(i+1)])

    # HNSW_data[HNSW_index:-1]    = Kmeans_data[array_index:Kmeans_data.shape[0]]
    # HNSW_data_ID[HNSW_index:-1] = Kmeans_data_ID[array_index:Kmeans_data.shape[0]]
    # HNSW_data[-1]    = centorid[i]
    # HNSW_data_ID[-1] = (1000000+i)
    # HNSW_data = np.float32(HNSW_data)

    # Kmeans_data = np.float32(Kmeans_data)
    # HNSWindex   = faiss.IndexHNSWFlat(d, maxN)
    # HNSWindex.hnsw.efConstruction = efCon
    # HNSWindex.hnsw.efSearch = eFSearch
    # HNSWindex.verbose = True
    # HNSWindex.add(HNSW_data)
    
    # faiss.write_index(HNSWindex, index_name.format(str(i)))

    # np.save(HNSW_save.format(str(i)), HNSW_data)
    # np.save(HNSW_ID_save.format(str(i)), HNSW_data_ID)

# PQ
bits  = 8
nlist = 2**bits
m     = 32

# quantizer = faiss.IndexFlatL2(d)  # the other index
PQtrain   = np.float32(PQ_train)

opq_index = faiss.ProductQuantizer(d, m, bits)
opq = faiss.OPQMatrix(d, m)
opq.pq = opq_index
opq.train(PQtrain)

for i in range (C):
    data = np.load(PQ_save.format(str(i)))
    data = np.float32(data)
    data = opq.apply_py(data)

    np.save(OPQ_save.format(str(i)), data)

query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query = np.float32(query)
query = opq.apply_py(query)
np.save("./input/HNSWOPQ_with_OPQ/OPQ_Query.npy", query)

OPQquantizer = faiss.IndexFlatL2(d)
PQtrain = opq.apply_py(PQtrain)
OPQindex = faiss.IndexIVFPQ(OPQquantizer, d, nlist, m, bits)
OPQindex.train(PQtrain)

OPQindex_name = "./input/HNSWOPQ_with_OPQ/OPQ_index/index_OPQ.index"
faiss.write_index(OPQindex, OPQindex_name)


# np.save("./input/HNSW_with_PQ/PQ_data/PQ_data.npy", PQ_train)
# np.save("./input/HNSW_with_PQ/PQ_data_ID/PQ_data_ID.npy", PQ_train_ID)

HNSWOPQ_data_path = "./input/HNSWOPQ_with_OPQ/HNSW_index/index_HNSW_{}.index"

m = 32

HNSW_OPQ_data = np.zeros((1000256-PQNN*C,d))
HNSW_OPQ_num = 0
for i in range (C):
    data = np.load(HNSW_save.format(str(i)))
    HNSW_OPQ_data[HNSW_OPQ_num:HNSW_OPQ_num+data.shape[0]] = data
    HNSW_OPQ_num += data.shape[0]

HNSW_OPQ_data = np.float32(HNSW_OPQ_data)
HNSW_opq_index = faiss.ProductQuantizer(d, m, bits)
HNSW_opq = faiss.OPQMatrix(d, m)
HNSW_opq.pq = HNSW_opq_index
HNSW_opq.train(HNSW_OPQ_data)

query = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query = np.float32(query)
query = HNSW_opq.apply_py(query)
np.save("./input/HNSWOPQ_with_OPQ/HNSW_Query.npy", query)

HNSW_OPQ_data = HNSW_opq.apply_py(HNSW_OPQ_data)

HNSWOPQquantizer = faiss.IndexFlatL2(d)
OPQindex = faiss.IndexIVFPQ(HNSWOPQquantizer, d, nlist, m, bits)
OPQindex.train(HNSW_OPQ_data)
compute_keys = True
for i in range(C):
    data = np.load(HNSW_save.format(str(i)))
    num = data.shape[0]
    data = HNSW_opq.apply_py(data)
    list_nos = np.zeros(num)
    list_nos = np.int64(list_nos)
    codes = np.empty((num,OPQindex.code_size), dtype=np.uint8)
    OPQindex.encode_multiple(num, faiss.swig_ptr(list_nos), faiss.swig_ptr(data), faiss.swig_ptr(codes),compute_keys )
    OPQindex.nprobe = nlist
    xcodes = np.empty((num,d), dtype=np.float32)
    OPQindex.decode_multiple(num, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes)) 

    HNSWOPQindex = faiss.IndexHNSWFlat(d, maxN)
    HNSWOPQindex.hnsw.efConstruction = efCon
    HNSWOPQindex.hnsw.efSearch = eFSearch
    HNSWOPQindex.verbose = True
    HNSWOPQindex.add(xcodes)

    faiss.write_index(HNSWOPQindex, HNSWOPQ_data_path.format(str(i)))

    

