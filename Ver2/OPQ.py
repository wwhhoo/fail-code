import numpy as np
import faiss
import time
from math import log10



PQ_weight_path  = "./input/OPQ_index/index_1M_OPQ.index"

data_path = './input/SIFT1M/SIFT1M_OPQ.npy'
data      = np.load(data_path)
data      = np.float32(data)
query     = np.load("./input/SIFT1M/SIFT1M_OPQ_Query.npy")
query     = np.float32(query)
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)


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

if __name__ == '__main__':

    bits  = 8
    nlist = 2**bits
    top = 1
    ad =  1000  
    
    index_ivfPQ = faiss.read_index(PQ_weight_path)
    index_ivfPQ.add(data)
    index_ivfPQ.nprobe = nlist
    time_cost = 0.0
    recall_score = np.zeros(4)
    for times in range(query.shape[0]):#
        single_query = query[times]
        single_query = np.reshape(single_query,(1,128))
        start_time = time.time()
        PQ_D, PQ_I = index_ivfPQ.search(single_query, 1000)
        # print(PQ_I.shape)
        # print(single_query)
        # print(label[times][0])
        end_time = time.time()
        time_cost += end_time-start_time

        recall_score += recall(label[times], PQ_I[0], top, ad)

    print(time_cost)  
    print(recall_score)