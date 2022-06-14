import numpy as np
import faiss
import time


d     = 128


d = 128                           # dimension
nb = 1000000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

nlist = 256
m = 16
k = 1000
for i in range(100000000000):

# quantizer = faiss.IndexFlatL2(d)  # this remains the same
# index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#                                   # 8 specifies that each sub-vector is encoded as 8 bits
# start_time = time.time()
# index.train(xb)
# end_time = time.time()
# print("Build index:",end_time - start_time)
# index.add(xb)
# index.nprobe = nlist
#######################
# maxN  = 16
# efCon = 200
# PQNN  = 512
# index   = faiss.IndexHNSWFlat(d, maxN)
# index.hnsw.efConstruction = efCon
# index.hnsw.efSearch = 65536
# index.verbose = True
# start_time = time.time()
# index.add(xb)
# end_time = time.time()
# print("Build index:",end_time - start_time)
#######################
start_time = time.time()
D, I = index.search(xq, k) # sanity check
end_time = time.time()
print("Search:",end_time - start_time)
