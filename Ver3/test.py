import numpy as np
import faiss
# d = 64                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.

# import faiss

# nlist = 64
# m = 8
# k = 4
# for i in range(0,0):
#     print("aaa")
# quantizer = faiss.IndexFlatL2(d)  # this remains the same
# index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#                                   # 8 specifies that each sub-vector is encoded as 8 bits
# index.train(xb)
# index.add(xb)
# compute_keys = True
# # Keys
# list_nos = np.zeros(xb.shape[0])
# for i in range(xb.shape[0]):
#     list_nos[i] = 18
# list_nos = np.int64(list_nos)
# # allocate memory
# codes = np.empty((nb,index.code_size), dtype=np.uint8)
# # encoder
# index.encode_multiple(nb, faiss.swig_ptr(list_nos), faiss.swig_ptr(xb), faiss.swig_ptr(codes),compute_keys )

# index.nprobe = 10              # make comparable with experiment above
# D, I = index.search(xq, k)     # search
# print(codes)
# xcodes = np.empty((nb,d), dtype=np.float32)
# # decoder
# index.decode_multiple(nb, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes)) 
# print(xcodes)
# avg_relative_error = ((xcodes - xb)**2).sum() / nb
# print(avg_relative_error)


OPQ_train         = np.load("./input/HNSWOPQ_with_OPQ_ver3/total_OPQ.npy")
OPQ_train = np.float32(OPQ_train)
HNSW_OPQ_train    = np.load("./input/HNSWOPQ_with_OPQ_ver3/total_HNSW.npy")
HNSW_OPQ_train = np.float32(HNSW_OPQ_train)
OPQ_train_ID      = np.load('./input/HNSWOPQ_with_OPQ_ver3/total_OPQ_ID.npy')
HNSW_OPQ_train_ID = np.load('./input/HNSWOPQ_with_OPQ_ver3/total_HNSW_ID.npy')
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)
query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query     = np.float32(query)
# OPQ_train = total_end50[:end_num]
# OPQ_train = np.float32(OPQ_train)
index = faiss.IndexFlatL2(128)
index.add(OPQ_train)
D,I = index.search(query,1)

score = 0
# OPQ_train_ID = total_end50_ID[:end_num]
for i in range(10000):
    if OPQ_train_ID[I[i,0]] ==  label[i,0]:
        score +=1
    # if score > max_recall:
    #     max_recall = score
        # max_k = knn
print(score)