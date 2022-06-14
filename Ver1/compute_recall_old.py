import numpy as np
import faiss
import time
from math import log10

query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query     = np.float32(query)
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)
Centroids = np.load("./input/kmeans_centroids.npy")
Centroids = np.float32(Centroids)




def PQ_cand(faiss_I, query, PQNN):
    PQ_weight_path = "./input/HNSW_with_PQ/PQ_index/index_PQ.index"
    PQ_data_path = "./input/HNSW_with_PQ/PQ_data/PQ_data_{}.npy"
    PQ_ID_path = "./input/HNSW_with_PQ/PQ_data_ID/PQ_data_ID_{}.npy"

    # d     = 128                           # dimension
    bits  = 8
    nlist = 2**bits
    # m     = 16

    PQ_data  = np.load(PQ_data_path.format(str(faiss_I)))
    PQ_data  = np.float32(PQ_data)
    PQ_ID    = np.load(PQ_ID_path.format(str(faiss_I)))
    PQ_ID    = np.int64(PQ_ID)
    PQ_ID    = np.reshape(PQ_ID, PQ_ID.shape[0])

    index_ivfPQ = faiss.read_index(PQ_weight_path)
    # must float 32, int 64
    index_ivfPQ.add_with_ids(PQ_data,PQ_ID)
    index_ivfPQ.nprobe = nlist
    PQ_start_time = time.time()
    PQ_D, PQ_I = index_ivfPQ.search(query, PQNN)
    PQ_end_time = time.time()
    time_cost = (PQ_end_time-PQ_start_time)

    return PQ_D, PQ_I, time_cost

def get_cnad( faiss_I, query, PQNN, k):
    HNSW_weight_path = "./input/HNSW_with_PQ/HNSW_index/index_HNSW_{}.index"
    HNSW_ID_path = "./input/HNSW_with_PQ/HNSW_data_ID/HNSW_data_ID_{}.npy"
    # maxN  = 16
    # efCon = 200
    total_D = np.asarray([])
    total_I = np.asarray([],dtype=int)
    HNSW_time = 0.0
    PQ_time = 0.0
    for i in faiss_I:
        HNSW_ID = np.load(HNSW_ID_path.format(str(i)))
        HNSW_ID = np.int64(HNSW_ID)

        HNSWindex = faiss.read_index(HNSW_weight_path.format(str(i)))

        query = np.reshape(query,(1,128))
        HNSW_start_time = time.time()
        HNSW_D, HNSW_I = HNSWindex.search(query,k)
        HNSW_end_time = time.time()
        HNSW_time = HNSW_time + HNSW_end_time-HNSW_start_time
        
        cand_ID = HNSW_ID[HNSW_I[0]]
        if max(cand_ID) >= 1000000:
            pos = np.where(cand_ID>=1000000)
            cand_ID = np.delete(cand_ID,pos,axis=None)
            HNSW_D = np.delete(HNSW_D,pos,axis=None)

        PQ_D, PQ_I, PQ_time_cost = PQ_cand(i,query, PQNN)
        PQ_time += PQ_time_cost
        total_D = np.concatenate((total_D,HNSW_D, PQ_D), axis=None)
        total_I = np.concatenate((total_I,cand_ID, PQ_I), axis=None)

        # else:

        #     total_D = np.concatenate((total_D,HNSW_D), axis=None)
        #     total_I = np.concatenate((total_I,cand_ID), axis=None)


    return total_D, total_I, HNSW_time, PQ_time


def sort_dis(total_D, total_I):
    # print(total_D[:10])
    sort_index = np.argsort(total_D)
    total_D = total_D[sort_index]
    total_I = total_I[sort_index]

    return total_D, total_I

def recall(label, faiss_I, top, ad):
    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)
    for top_num in range(top):
        # print(faiss_I[0].shape)
        hit = np.where(faiss_I == label[top_num])
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

# def recall(label, faiss_I, top):

#     hit = 0
#     # print(label)
#     # print(faiss_I)
#     for top_num in range(top):
#         index = np.where(faiss_I == label[top_num])
#         # print(index)
#         if index[0].size > 0:
#             hit += 1

#     return hit


if __name__ == '__main__':

    d     = 128                           # dimension
    bits  = 8
    nlist = 2**bits
    m     = 16
    k     = 10
    top   = 1
    PQNN  = 100
    C     = 256
    clu   = 5
    ad    = 1000

    start_time = time.time()
    flat_index = faiss.IndexFlatL2(d)
    flat_index.add(Centroids)
    D, I = flat_index.search(query, clu)

    total_HNSW_time_cost = 0.0
    total_PQ_time_cost = 0.0
    recall_score = np.zeros(4)

    # recall_score = 0
    for i in range(query.shape[0]):

        total_D, total_I, HNSW_cost, PQ_cost = get_cnad(I[i], query[i], PQNN, k)
        total_D, total_I = sort_dis(total_D,total_I)
        recall_score += recall(label[i], total_I, top, ad)
        total_HNSW_time_cost += HNSW_cost
        total_PQ_time_cost += PQ_cost


    end_time = time.time()
    time_taken = end_time - start_time
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)
    print("This took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 
    recall_score = (recall_score/(query.shape[0]*top))
    print(recall_score)
    np.savetxt("./output/HNSW_PQ_recall.txt", recall_score)
    print("HNSW search time:", total_HNSW_time_cost)
    print("PQ search time:", total_PQ_time_cost)
    print("Search time:",total_HNSW_time_cost+total_PQ_time_cost)
    
