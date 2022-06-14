import numpy as np
import faiss
import time

data_path = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
# ID_path = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
d = 128
maxN = 16
efCon = 200
eFSearch = 100
total_time = 0.0
for i in range (256):

    first_stage = np.load(data_path.format(str(i)))
    # first_stage_ID = np.load(ID_path.format(str(i)))
    first_stage = np.float32(first_stage)
    # first_stage_ID = np.int64(first_stage_ID)
    index = faiss.IndexHNSWFlat(d, maxN)
    index.hnsw.efConstruction = efCon
    index.hnsw.efSearch = eFSearch
    index.verbose = True
    start_time = time.time()
    index.add(first_stage)
    end_time = time.time()
    total_time += (end_time-start_time)
    index_name = "./input/HNSW_index/index_HNSW_{}.index".format(str(i))
    faiss.write_index(index, index_name)
print(total_time)