import numpy as np
import faiss


if __name__ == '__main__':

    Kmeans_path        = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
    Kmeans_ID_path     = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
    Sec_Kmeans_path    = './input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256/Sec_Kmeans{}_{}.npy'
    Sec_Kmeans_ID_path = './input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID/Sec_Kmeans{}_{}_ID.npy'
    Sec_HNSW_path      = './input/HNSWOPQ_with_OPQ_ver3/Sec_Level_HNSW_data/Sec_HNSW_{}.npy'
    Sec_HNSW_ID_path   = './input/HNSWOPQ_with_OPQ_ver3/Sec_Level_HNSW_dataID/Sec_HNSW_{}_ID.npy'
    centorid_path      = "./input/HNSWOPQ_with_OPQ_ver3/centroids/kmeans_centroids_{}.npy"
    # data_graph_path    = "./input/result/result{}/data_graph_{}/data_graph_{}_PageRank.txt"
    # data_graph_path    = "./input/result/result{}/data_graph_{}/data_graph_{}_HITS_hub.txt"
    data_graph_path    = "./input/result/result{}/data_graph_{}/data_graph_{}_HITS_authority.txt"

    d   = 128
    knn = 16
    C   = 256
    SC  = 256

    FlatIndex = faiss.IndexFlatL2(d)
    total_top50 = np.zeros((1000000,d),dtype=int)
    total_end50 = np.zeros((1000000,d),dtype=int)
    total_top50_ID = np.zeros((1000000),dtype=int)
    total_end50_ID = np.zeros((1000000),dtype=int)
    top_num = 0
    end_num = 0
    for times in range (C):
        dataset   = np.load(Kmeans_path.format(str(times)))
        datasetID = np.load(Kmeans_ID_path.format(str(times)))

        dataset = np.float32(dataset)

        
        # res = faiss.StandardGpuResources()
        score = np.loadtxt(data_graph_path.format(str(int(times/32)),str(times),str(times)))
        # data_graph_index = 0
        # FlatIndex.add(dataset)
        # D, I = FlatIndex.search(dataset, knn)
        # FlatIndex.reset()
        # score = np.zeros(dataset.shape[0],dtype=int)

        # for fir in range (I.shape[0]):
        #     for sec in range (knn):
        #         score[I[fir, sec]] += 1
                # data_graph[data_graph_index] = I[fir, sec]
                # data_graph_index += 1

        # np.save(data_graph_path.format(str(times)), I)
        
        sort_ID = np.argsort(score)

        top_50 = dataset[sort_ID[:int(dataset.shape[0]/2)]]
        end_50 = dataset[sort_ID[int(dataset.shape[0]/2):]]
        top_50_ID = datasetID[sort_ID[:int(datasetID.shape[0]/2)]]
        end_50_ID = datasetID[sort_ID[int(datasetID.shape[0]/2):]]

        total_top50[top_num:top_num+top_50.shape[0]] = top_50
        total_end50[end_num:end_num+end_50.shape[0]] = end_50
        total_top50_ID[top_num:top_num+top_50.shape[0]] = top_50_ID
        total_end50_ID[end_num:end_num+end_50.shape[0]] = end_50_ID
        top_num += top_50.shape[0]
        end_num += end_50.shape[0]

        np.save(Sec_HNSW_path.format(str(times)),top_50)
        np.save(Sec_HNSW_ID_path.format(str(times)),top_50_ID)

        cent = SC
        if end_50.shape[0] < SC :
            cent = end_50.shape[0]

        ncentroids = cent
        niter      = 200
        verbose    = True
        kmeans     = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
        kmeans.train(end_50)

        kmeans.nprobe = niter
        D, I = kmeans.index.search(end_50, 1)

        count       = np.zeros(SC,dtype=int)
        output_data = np.zeros((SC,10000,d),dtype=np.uint8)
        data_ID     = np.zeros((SC,10000),dtype=np.uint32)
        end_50      = np.uint8(end_50)

        for i in range(end_50.shape[0]):
   
            output_data[I[i],count[I[i]]] = end_50[i]
            data_ID[I[i],count[I[i]]]     = end_50_ID[i]
            count[I[i]] += 1

        for i in range(output_data.shape[0]):
            np.save(Sec_Kmeans_path.format(str(times),str(i)), output_data[i,:count[i]])
            np.save(Sec_Kmeans_ID_path.format(str(times),str(i)), data_ID[i,:count[i]])
        np.save(centorid_path.format(str(times)),kmeans.centroids)
    # print(num)
    np.save('./input/HNSWOPQ_with_OPQ_ver3/total_HNSW.npy',total_top50[:top_num])
    np.save('./input/HNSWOPQ_with_OPQ_ver3/total_OPQ.npy',total_end50[:end_num])
    np.save('./input/HNSWOPQ_with_OPQ_ver3/total_HNSW_ID.npy',total_top50_ID[:top_num])
    np.save('./input/HNSWOPQ_with_OPQ_ver3/total_OPQ_ID.npy',total_end50_ID[:end_num])