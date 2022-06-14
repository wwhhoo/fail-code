import numpy as np
import faiss
import time
from math import log10


data_path = './input/SIFT1M/SIFT1M.npy'
data      = np.load(data_path)
data      = np.float32(data)
query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query     = np.float32(query)
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)

d = 128
maxN = 4
efCon = 200
eFSearch = 100
total_time = 0.0
k = 1
top = 1
ad = 10

def recall(label, faiss_I, top, ad):
    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)
    for top_num in range(top):
        # print(faiss_I[0].shape)
        hit = np.where(faiss_I[0] == label[top_num])
        if hit[0].size > 0:
            if (hit[0] < 1 and top == 1):
                hit_rate[0:int(log10(ad)+1)] += 1
            # elif (hit[0] < 10 and top <= 10 ):
            #     hit_rate[1:int(log10(ad)+1)] += 1
            # elif (hit[0] < 100 ):
            #     hit_rate[2:int(log10(ad)+1)] += 1
            # elif (hit[0] < 1000 ):
            #     hit_rate[3] += 1
    return hit_rate

index = faiss.IndexHNSWFlat(d, maxN)
index.hnsw.efConstruction = efCon
index.hnsw.efSearch = eFSearch
# index.hnsw.max_level = 10
index.verbose = True
start_time = time.time()
index.add(data)
end_time = time.time()
total_time = (end_time-start_time)
faiss.write_index(index, "HNSW1M.index")
print(total_time)
# index = faiss.read_index("HNSW1M.index")


HNSW_start_time = time.time()
HNSW_D, HNSW_I = index.search(query,k)
HNSW_end_time = time.time()
total_time = (HNSW_end_time-HNSW_start_time)
time_taken = HNSW_end_time - HNSW_start_time
hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)
print("This took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 
recall_score    = np.zeros(2)
for i in range (query.shape[0]):
    recall_score += recall(label[i], HNSW_I[i], top, ad)


print(recall_score)