import numpy as np
import faiss


d     = 128
bits  = 8
nlist = 2**bits
m     = 32

data = np.load("./input/SIFT1M/SIFT1M.npy")

quantizer = faiss.IndexFlatL2(d)  # the other index
PQtrain   = np.float32(data)

opq_index = faiss.ProductQuantizer(d, m, bits)
opq = faiss.OPQMatrix(d, m)
opq.pq = opq_index
opq.train(PQtrain)

query = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query = np.float32(query)
query = opq.apply_py(query)
np.save("./input/SIFT1M/SIFT1M_OPQ_Query.npy",query)

PQtrain = opq.apply_py(PQtrain)
np.save("./input/SIFT1M/SIFT1M_OPQ.npy",PQtrain)

PQ_data_path    = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
OPQ_data_path    = './input/SIFT1M_OPQ_Kmeans256/SIFT1M_OPQ_Kmeans256_{}.npy'
for i in range(256):
    PQdata = np.load(PQ_data_path.format(str(i)))
    PQdata = np.float32(PQdata)
    OPQdata = opq.apply_py(PQdata)
    np.save(OPQ_data_path.format(str(i)),OPQdata)

PQindex   = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
PQindex.train(PQtrain)
# PQindex_name = "./input/PQ_index/index_1M_PQ.index"
PQindex_name = "./input/OPQ_index/index_1M_OPQ.index"
faiss.write_index(PQindex, PQindex_name)