import numpy as np
import faiss
import time
from math import log10



PQ_weight_path  = "./input/PQ_index/index_1M_PQ.index"

data_path = './input/SIFT1M/SIFT1M.npy'
data      = np.load(data_path)
data      = np.float32(data)
query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query     = np.float32(query)
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)
top = 1
ad =  1000

def recall(label, faiss_I, top, ad):
    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)
    for top_num in range(top):

        hit = np.where(faiss_I == label[top_num])
        if hit[0].size > 0:
            if (hit[0] < 1 and top == 1):
                hit_rate[0:int(log10(ad)+1)] += 1
            elif (hit[0] < 10 and top <= 10 ):
                hit_rate[1:int(log10(ad)+1)] += 1
            elif (hit[0] < 100 ):
                hit_rate[2:int(log10(ad)+1)] += 1
            elif (hit[0] < 1000 ):
                hit_rate[3] += 1
    return hit_rate


bits  = 8
nlist = 2**bits

index_ivfPQ = faiss.read_index(PQ_weight_path)
index_ivfPQ.add(data)
index_ivfPQ.nprobe = nlist
start_time = time.time()
PQ_D, PQ_I = index_ivfPQ.search(query, 1000)
end_time = time.time()
time_cost = end_time-start_time

print(time_cost)

recall_score    = np.zeros(4)
for i in range(query.shape[0]):

    recall_score    += recall(label[i], PQ_I[i], top, ad)
print(recall_score)