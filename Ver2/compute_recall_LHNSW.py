import numpy as np
import faiss
import time
from math import log10

query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
query     = np.float32(query)
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)
Centroids = np.load("./input/kmeans_centroids.npy")
Centroids = np.float32(Centroids)



def get_cand( faiss_I, query, k):
    HNSW_weight_path = "./input/HNSW_index/index_HNSW_{}.index"
    HNSW_ID_path = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
    maxN  = 16
    efCon = 200
    total_D = np.asarray([])
    total_I = np.asarray([],dtype=int)

    total_time = 0.0
    
    for i in faiss_I:
        HNSW_ID = np.load(HNSW_ID_path.format(str(i)))
        HNSW_ID = np.int64(HNSW_ID)
        HNSWindex = faiss.read_index(HNSW_weight_path.format(str(i)))

        query = np.reshape(query,(1,128))
        start_time = time.time()
        HNSW_D, HNSW_I = HNSWindex.search(query,k)
        end_time = time.time()
        total_time += (end_time-start_time)
        cand_ID = HNSW_ID[HNSW_I[0]]
      
        total_D = np.concatenate((total_D,HNSW_D), axis=None)
        total_I = np.concatenate((total_I,cand_ID), axis=None)
        
    return total_D, total_I, total_time

def sort_dis(total_D, total_I):
    # print(total_D[:10])
    sort_index = np.argsort(total_D)
    total_D = total_D[sort_index]
    total_I = total_I[sort_index]

    return total_D, total_I

def recall(label, faiss_I, top, ad, query):
    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)
    for top_num in range(top):
        # print(faiss_D.shape)
        hit = np.where(faiss_I == label[top_num])
        if hit[0].size > 0:
            if (hit[0] < 1 and top == 1):
                hit_rate[0:int(log10(ad)+1)] += 1
            # elif (hit[0] < 10 and hit[0] != 0 ):
            #     print(query)
            #     print(faiss_I[:10])
            elif (hit[0] < 10 and top <= 10 ):
                hit_rate[1:int(log10(ad)+1)] += 1
            elif (hit[0] < 100 ):
                hit_rate[2:int(log10(ad)+1)] += 1
            elif (hit[0] < 1000000 ):
                hit_rate[3] += 1
    return hit_rate

if __name__ == '__main__':

    d    = 128
    k    = 10
    top  = 1
    clu  = 0
    ad   = 1000
    total_time = 0.0


    for times in range (4):
        clu+=5
        flat_index = faiss.IndexFlatL2(d)
        flat_index.add(Centroids)
        D, I = flat_index.search(query, clu)
        
        start_time = time.time()

        recall_score = np.zeros(4)
        for i in range(query.shape[0]):
            # print(i)
            total_D, total_I, time_cost = get_cand(I[i], query[i], k)
            total_D, total_I = sort_dis(total_D,total_I)
            recall_score += recall(label[i], total_I, top, ad, query[i])
            total_time += time_cost

            # break

        end_time = time.time()
        time_taken = end_time - start_time
        hours, rest = divmod(time_taken,3600)
        minutes, seconds = divmod(rest, 60)
        print("This took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 
        recall_score = (recall_score/(query.shape[0]*top))
        print(recall_score)
        np.savetxt("./output/HNSW_recall.txt", recall_score)
        print(total_time)   
    

