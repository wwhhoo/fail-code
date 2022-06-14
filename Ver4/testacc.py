import numpy as np
import faiss

OPQ_train         = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/total_OPQ.npy")
OPQ_train = np.float32(OPQ_train)
HNSW_OPQ_train    = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/totalO_HNSW.npy")
HNSW_OPQ_train = np.float32(HNSW_OPQ_train)
OPQ_train_ID      = np.load('./input/HNSWOPQ_with_OPQ_ver3.5/total_OPQ_ID.npy')
HNSW_OPQ_train_ID = np.load('./input/HNSWOPQ_with_OPQ_ver3.5/total_HNSW_ID.npy')
OPQ_data_path = "./input/HNSWOPQ_with_OPQ_ver3.5/OPQ32_data/Sec_OPQ_data_{}_{}.npy"
OPQ_ID_path = "./input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID/Sec_Kmeans{}_{}_ID.npy"

# Kmeans_path        = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
# Kmeans_ID_path     = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
# Sec_Kmeans_path    = './input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256/Sec_Kmeans{}_{}.npy'
# Sec_Kmeans_ID_path = './input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID/Sec_Kmeans{}_{}_ID.npy'
# Sec_HNSW_path      = './input/HNSWOPQ_with_OPQ_ver3/Sec_Level_HNSW_data/Sec_HNSW_{}.npy'
# Sec_HNSW_ID_path   = './input/HNSWOPQ_with_OPQ_ver3/Sec_Level_HNSW_dataID/Sec_HNSW_{}_ID.npy'
# centorid_path      = "./input/HNSWOPQ_with_OPQ_ver3/centroids/kmeans_centroids_{}.npy"

# query     = np.loadtxt("./input/SIFT1M/SIFT1M_Query.txt")
# query     = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/OPQ_Query.npy")
query     = np.load("./input/HNSWOPQ_with_OPQ_ver3.5/HNSW_Query.npy")
query     = np.float32(query)
label     = np.loadtxt("./input/SIFT1M/SIFT1M_Groundtruth_100NN.txt", dtype=int)


# for i in range ()
max_recall = 0
max_k = 0
knn = 100
# for j in range (100):
# print(OPQ_train)

#     d   = 128
#     knn +=1
#     C   = 256
#     SC  = 256

#     FlatIndex = faiss.IndexFlatL2(d)
#     total_top50 = np.zeros((1000000,d),dtype=int)
#     total_end50 = np.zeros((1000000,d),dtype=int)
#     total_top50_ID = np.zeros((1000000),dtype=int)
#     total_end50_ID = np.zeros((1000000),dtype=int)
#     top_num = 0
#     end_num = 0
#     for times in range (C):
#         dataset   = np.load(Kmeans_path.format(str(times)))
#         datasetID = np.load(Kmeans_ID_path.format(str(times)))

#         dataset = np.float32(dataset)

        
#         # res = faiss.StandardGpuResources()
#         FlatIndex.add(dataset)
#         D, I = FlatIndex.search(dataset, knn)
#         FlatIndex.reset()
#         score = np.zeros(dataset.shape[0],dtype=int)
#         for fir in range (I.shape[0]):
#             for sec in range (knn):
#                 score[I[fir, sec]] += 1
        
#         sort_ID = np.argsort(score)

#         top_50 = dataset[sort_ID[:int(dataset.shape[0]/2)]]
#         end_50 = dataset[sort_ID[int(dataset.shape[0]/2):]]
#         top_50_ID = datasetID[sort_ID[:int(datasetID.shape[0]/2)]]
#         end_50_ID = datasetID[sort_ID[int(datasetID.shape[0]/2):]]

#         total_top50[top_num:top_num+top_50.shape[0]] = top_50
#         total_end50[end_num:end_num+end_50.shape[0]] = end_50
#         total_top50_ID[top_num:top_num+top_50.shape[0]] = top_50_ID
#         total_end50_ID[end_num:end_num+end_50.shape[0]] = end_50_ID
#         top_num += top_50.shape[0]
#         end_num += end_50.shape[0]

#     OPQ_train = total_end50[:end_num]
#     OPQ_train = np.float32(OPQ_train)
index = faiss.IndexFlatL2(128)
index.add(HNSW_OPQ_train)
D,I = index.search(query,1)

score = 0
HNSW_OPQ_train_ID = HNSW_OPQ_train_ID#total_end50_ID[:end_num]
for i in range(10000):
    if HNSW_OPQ_train_ID[I[i,0]] ==  label[i,0]:
        score +=1
        # if score > max_recall:
        #     max_recall = score
        #     max_k = knn
print(score)
    # if j % 10 == 9:

    #     print(max_recall)
    #     print(max_k)