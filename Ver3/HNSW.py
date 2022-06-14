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

    bits  = 8
    nlist = 2**bits
    m     = 16

    d = 128
    maxN = 16
    efCon = 200
    eFSearch = 100
    total_time = 0.0
    k = 100
    top = 1
    ad = 1000


    SIFT1M_data = np.float32(data)
    opq_index = faiss.ProductQuantizer(d, m, bits)
    opq = faiss.OPQMatrix(d, m)
    opq.pq = opq_index
    opq.train(SIFT1M_data)

    query = opq.apply_py(query)
    SIFT1M_data = opq.apply_py(SIFT1M_data)

    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    OPQindex = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    # 8 specifies that each sub-vector is encoded as 8 bits
    OPQindex.train(SIFT1M_data)

    numbers = SIFT1M_data.shape[0]
    compute_keys = True
    list_nos = np.zeros(numbers)
    list_nos = np.int64(list_nos)
    codes = np.empty((numbers,OPQindex.code_size), dtype=np.uint8)
    OPQindex.encode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(SIFT1M_data), faiss.swig_ptr(codes),compute_keys )
    OPQindex.nprobe = nlist              # make comparable with experiment above
    xcodes = np.empty((numbers,d), dtype=np.float32)
    OPQindex.decode_multiple(numbers, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes)) 


    index = faiss.IndexHNSWFlat(d, maxN)
    index.hnsw.efConstruction = efCon
    index.hnsw.efSearch = eFSearch
    index.verbose = True
    # start_time = time.time()
    index.add(xcodes)
    # end_time = time.time()
    # total_time = (end_time-start_time)
    faiss.write_index(index, "HNSW1M.index")
    # print(total_time)
    # index = faiss.read_index("HNSW1M.index")
    time_taken = 0.0
    recall_score    = np.zeros(4)
    for times in range (query.shape[0]):
        simgle_query = query[times]
        simgle_query = np.reshape(simgle_query,(1,128))
        HNSW_start_time = time.time()
        HNSW_D, HNSW_I = index.search(simgle_query,k)
        HNSW_end_time = time.time()
        time_taken += (HNSW_end_time - HNSW_start_time)

        recall_score += recall(label[times], HNSW_I, top, ad)

    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)
    print("This took %d hours %d minutes %f seconds" %(hours,minutes,seconds))


print(recall_score/10000)