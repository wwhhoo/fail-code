import numpy as np
import faiss
import time
from math import log10


def PQ_cand(PQ_weight_path, PQ_data, PQ_ID, query, PQNN):

    
    bits  = 8
    nlist = 2**bits
    PQ_data  = np.float32(PQ_data)
    PQ_ID    = np.int64(PQ_ID)
    # PQ_ID    = np.reshape(PQ_ID, PQ_ID.shape[0])

    index_ivfPQ = faiss.read_index(PQ_weight_path)
    # must float 32, int 64
    index_ivfPQ.add_with_ids(PQ_data,PQ_ID)
    # index_ivfPQ.add(PQ_data)
    index_ivfPQ.nprobe = nlist
    PQ_start_time = time.time()
    PQ_D, PQ_I = index_ivfPQ.search(query, PQNN)
    PQ_end_time = time.time()
    time_cost = (PQ_end_time-PQ_start_time)

    return PQ_D, PQ_I, time_cost

def get_cand( HNSWquery, OPQquery, PQNN, OPQ_data, OPQ_ID, HNSW_OPQ_data, HNSW_OPQ_ID ):

    OPQ32_weight_path = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_index/index_OPQ.index"
    OPQ16_weight_path = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_index/HNSW_OPQindex.index"

    candidate = 0
    
    total_D = np.asarray([])
    total_I = np.asarray([],dtype=int)
    

    # OPQquery = np.reshape(OPQquery,(1,128))
    # HNSWquery = np.reshape(HNSWquery,(1,128))
    HNSW_time = 0.0
    PQ_time = 0.0 
    num = 0

    PQ_D, PQ_I, PQ_time_cost = PQ_cand(OPQ32_weight_path, OPQ_data, OPQ_ID, OPQquery, PQNN)
    total_D = PQ_D
    total_I = PQ_I
    # PQ_time = PQ_time_cost
    PQ_D, PQ_I, PQ_time_cost = PQ_cand(OPQ16_weight_path, HNSW_OPQ_data, HNSW_OPQ_ID, HNSWquery, PQNN)
    # total_D = PQ_D
    # total_I = PQ_I
    # PQ_D = np.float64(PQ_D)
    total_D = np.concatenate((total_D,PQ_D),axis=1)
    total_I = np.concatenate((total_I,PQ_I),axis=1)

    PQ_time += PQ_time_cost


    return total_D, total_I, HNSW_time, PQ_time, candidate, num


def sort_dis(total_D, total_I):
    sort_D = np.zeros((total_D.shape))
    sort_I = np.zeros((total_I.shape))
    for i in range (total_I.shape[0]):
            
        sort_index = np.argsort(total_D[i])
        sort_D[i] = total_D[i][sort_index]
        sort_I[i] = total_I[i][sort_index]

    return sort_D, sort_I

def recall(label, faiss_I, top, ad):
    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)
    
    for top_num in range(top):
        
        hit = np.where(faiss_I == label[top_num])
        # print(faiss_I.shape)
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

    d     = 128                           # dimension
    bits  = 8
    nlist = 2**bits
    m     = 16
    k     = 0
    top   = 1
    PQNN  = 100
    C     = 256
    clu   = 5
    ad    = 1000

    query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
    query     = np.float32(query)
    OPQquery  = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_Query.npy")
    OPQquery  = np.float32(OPQquery)
    HNSWquery = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/HNSW_Query.npy")
    HNSWquery = np.float32(HNSWquery)
    label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)
    OPQ_train       = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/totalO_OPQ.npy")
    HNSW_OPQ_train  = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/totalO_HNSW.npy")
    HNSW_OPQ_ID     = np.load('./input/HNSWOPQ_with_OPQ_ver3.5/total_HNSW_ID.npy')
    OPQ_ID          = np.load('./input/HNSWOPQ_with_OPQ_ver3.5/total_OPQ_ID.npy')
    


    for times in range(1):
        # clu +=5
        OPQnum = 0
        # k+=5
        # print(k)
        total_candidate = 0
        start_time = time.time()

        total_HNSW_time_cost = 0.0
        total_PQ_time_cost = 0.0
        recall_score = np.zeros(4)
        total_D, total_I, HNSW_cost, PQ_cost, candidate, num = get_cand( HNSWquery, OPQquery, PQNN, OPQ_train, OPQ_ID, HNSW_OPQ_train, HNSW_OPQ_ID)
        total_D, total_I = sort_dis(total_D,total_I)
        for i in range(query.shape[0]):
            
            recall_score += recall(label[i], total_I[i], top, ad)
            total_HNSW_time_cost += HNSW_cost
            total_PQ_time_cost += PQ_cost
            total_candidate += candidate
            OPQnum += num

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
        # print("Total candidate:", total_candidate/10000)
        # print("OPQ point:", OPQnum/10000/clu/k)

