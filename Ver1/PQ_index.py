import numpy as np
import faiss


d     = 128
bits  = 8
nlist = 2**bits
m     = 16

data = np.load("./input/SIFT1M/SIFT1M.npy")

quantizer = faiss.IndexFlatL2(d)  # the other index
PQtrain   = np.float32(data)

query = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query = np.float32(query)

PQindex   = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
PQindex.train(PQtrain)
PQindex_name = "./input/PQ_index/index_1M_PQ.index"

faiss.write_index(PQindex, PQindex_name)