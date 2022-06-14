import numpy as np
import faiss
import time
from math import log10

query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query     = np.float32(query)
OPQquery  = np.load("./input/SIFT1M/SIFT1M_OPQ_Query.npy")
OPQquery  = np.float32(OPQquery)
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)
Centroids = np.load("./input/kmeans_centroids.npy")
Centroids = np.float32(Centroids)




def OPQ_cand(faiss_I, query, PQNN, d):
    OPQ_weight_path  = "./input/OPQ_index/index_1M_OPQ.index"
    OPQ_data_path    = './input/SIFT1M_OPQ_Kmeans256/SIFT1M_OPQ_Kmeans256_{}.npy'
    OPQ_ID_path      = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'

    
    # d     = 128                           # dimension
    bits  = 8
    nlist = 2**bits
    # m     = 16
    query = np.reshape(query,(1,d))
    total_Data = np.asarray([])
    total_ID   = np.asarray([],dtype=int)
    test_time = 0.0
    for i in faiss_I:
        OPQ_data    = np.load(OPQ_data_path.format(str(i)))
        OPQ_data    = np.float32(OPQ_data)
        # print (OPQ_data.shape)
        OPQ_ID      = np.load(OPQ_ID_path.format(str(i)))
        OPQ_ID      = np.int64(OPQ_ID)
        OPQ_ID      = np.reshape(OPQ_ID, OPQ_ID.shape[0])

        if  i == faiss_I[0]:
            total_Data = OPQ_data
            total_ID   = OPQ_ID
        else:
            total_Data = np.concatenate((total_Data,OPQ_data))
            total_ID   = np.concatenate((total_ID,OPQ_ID))
    total_Data    = np.float32(total_Data)

    
    index_ivfPQ = faiss.read_index(OPQ_weight_path)
    index_ivfPQ.add_with_ids(total_Data,total_ID)
    index_ivfPQ.nprobe = nlist
    start_time = time.time()
    PQ_D, PQ_I = index_ivfPQ.search(query, PQNN)
    end_time = time.time()
    time_cost = (end_time-start_time)

    return PQ_D, PQ_I, time_cost, test_time

def recall(label, faiss_I, top, ad):
    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)
    for top_num in range(top):
        # print(faiss_I[0].shape)
        hit = np.where(faiss_I[0] == label[top_num])
        if hit[0].size > 0:
            if (hit[0] < 1 and top == 1):
                hit_rate[0:int(log10(ad)+1)] += 1
            elif (hit[0] < 10 and top <= 10 ):
                hit_rate[1:int(log10(ad)+1)] += 1
            elif (hit[0] < 100 ):
                hit_rate[2:int(log10(ad)+1)] += 1
            elif (hit[0] < 1000000 ):
                hit_rate[3] += 1
    return hit_rate


if __name__ == '__main__':

    d     = 128 
    bits  = 8
    nlist = 2**bits
    m     = 16
    k     = 100
    top   = 1
    PQNN  = 100
    C     = 256
    clu   = 5
    ad    = 1000

    start_time = time.time()

    flat_index = faiss.IndexFlatL2(d)
    flat_index.add(Centroids)
    D, I = flat_index.search(query, clu)

    recall_score    = np.zeros(4)
    total_time_cost = 0.0
    test_time_cost  = 0.0
    for i in range(query.shape[0]):
        total_D, total_I, time_cost, test_time = OPQ_cand(I[i], OPQquery[i], PQNN, d)
        total_time_cost += time_cost
        test_time_cost  += test_time
        recall_score    += recall(label[i], total_I, top, ad)
        # break

    end_time    = time.time()
    time_taken  = end_time - start_time
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)
    print("This took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 
    recall_score = recall_score/(query.shape[0]*top)
    print(recall_score)
    np.savetxt("./output/OPQ_recall.txt", recall_score)

    print("Search time:",total_time_cost)
    print("test Search time:",test_time_cost)
