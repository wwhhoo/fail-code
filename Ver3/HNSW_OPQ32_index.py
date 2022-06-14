import numpy as np
import faiss
import time

# data path load: Kmeans save: HNSW
SIFT1M_data = np.load("./input/SIFT1M/SIFT1M.npy")
data_path    = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
index_name   = "./input/HNSW_OPQ16/HNSW_OPQ16_index/index_HNSW_{}.index"
HNSW_save    = "./input/HNSW_OPQ16/HNSW_OPQ16_data/HNSW_data_{}.npy"
query = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query = np.float32(query)

d     = 128
maxN  = 16
efCon = 200
index = 0
C     = 256
eFSearch = 100


bits  = 8
nlist = 2**bits
m     = 32


SIFT1M_data = np.float32(SIFT1M_data)
opq_index = faiss.ProductQuantizer(d, m, bits)
opq = faiss.OPQMatrix(d, m)
opq.pq = opq_index
opq.train(SIFT1M_data)

query = opq.apply_py(query)
np.save("./input/HNSW_OPQ16/HNSW_OPQ16_Query.npy",query)

SIFT1M_data = opq.apply_py(SIFT1M_data)

quantizer = faiss.IndexFlatL2(d)  # this remains the same
OPQindex = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
OPQindex.train(SIFT1M_data)

for i in range(C):
    data = np.load(data_path.format(str(i)))
    data = np.float32(data)
    data = opq.apply_py(data)
    # OPQindex.reset()
    # OPQindex.add(data)
    numbers = data.shape[0]
    compute_keys = True
    # Keys
    list_nos = np.zeros(numbers)
    list_nos = np.int64(list_nos)

    # allocate memory
    codes = np.empty((numbers,OPQindex.code_size), dtype=np.uint8)
    # encoder
    OPQindex.encode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(data), faiss.swig_ptr(codes),compute_keys )

    OPQindex.nprobe = nlist              # make comparable with experiment above
    # D, I = index.search(xq, k)     # search
    # print(codes)
    xcodes = np.empty((numbers,d), dtype=np.float32)
    # decoder
    OPQindex.decode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes)) 
    # np.save(HNSW_save.format(str(i)),xcodes)
    # print(xcodes.shape)

    HNSWindex = faiss.IndexHNSWFlat(d, maxN)
    HNSWindex.hnsw.efConstruction = efCon
    HNSWindex.hnsw.efSearch = eFSearch
    HNSWindex.verbose = True
    start_time = time.time()
    HNSWindex.add(xcodes)
    end_time = time.time()
    total_time = (end_time-start_time)
    faiss.write_index(HNSWindex, index_name.format(str(i)))
    # print(total_time)

