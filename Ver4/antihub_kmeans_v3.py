import numpy as np
import faiss


def Sec_kmeans(data, data_ID, SC):
    
    cent = SC
    if data.shape[0] < SC :
        cent = data.shape[0]

    ncentroids = cent
    niter      = 200
    verbose    = True
    nredo      = 10
    kmeans     = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True, nredo=nredo)
    kmeans.train(data)
    kmeans.nprobe = niter
    D, I = kmeans.index.search(data, 1)

    count          = np.zeros(SC,dtype=int)
    kmeans_data    = np.zeros((SC,data.shape[0],d),dtype=np.uint8)
    kmeans_data_ID = np.zeros((SC,data.shape[0]),dtype=np.uint32)
    data           = np.uint8(data)
    Sec_clu_num    = np.zeros(SC,dtype=int)

    for i in range(data.shape[0]):
        kmeans_data[I[i],count[I[i]]]    = data[i]
        kmeans_data_ID[I[i],count[I[i]]] = data_ID[i]
        count[I[i]] += 1
    output_data = np.zeros((data.shape[0],data.shape[1]))
    output_data_ID = np.zeros((data.shape[0]),dtype=np.uint32)
    output_index = 0
    for fir in range(SC):
        Sec_clu_num[fir] = count[fir]
        output_data[output_index:output_index+count[fir]]    = kmeans_data[fir][:count[fir]]
        output_data_ID[output_index:output_index+count[fir]] = kmeans_data_ID[fir][:count[fir]]
        output_index += count[fir]
    

    return output_data, output_data_ID, kmeans.centroids, Sec_clu_num

if __name__ == '__main__':

    Kmeans_path        = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
    Kmeans_ID_path     = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
    OPQ_save_path      = "./input/HNSWOPQ_with_OPQ_ver4/Kmeans_high_data/Sec_OPQ_data_{}.npy"
    OPQ_ID_save_path   = "./input/HNSWOPQ_with_OPQ_ver4/Kmeans_high_ID/Sec_OPQ_{}_ID.npy"
    Sec_HNSW_path      = './input/HNSWOPQ_with_OPQ_ver4/Kmeans_low_data/Sec_HNSW_{}.npy'
    Sec_HNSW_ID_path   = './input/HNSWOPQ_with_OPQ_ver4/Kmeans_low_ID/Sec_HNSW_{}_ID.npy'
    OPQ_centorid_path  = "./input/HNSWOPQ_with_OPQ_ver4/centroids/centroids_OPQ32/kmeans_centroids_{}.npy"
    HNSW_centorid_path = "./input/HNSWOPQ_with_OPQ_ver4/centroids/centroids_OPQ16/kmeans_centroids_{}.npy"
    centorid_path      = "./input/HNSWOPQ_with_OPQ_ver4/centroids/kmeans_centroids_{}.npy"
    Sec_clu_num_H_path = "./input/HNSWOPQ_with_OPQ_ver4/Second_cluster_data_num_H.npy"
    Sec_clu_num_L_path = "./input/HNSWOPQ_with_OPQ_ver4/Second_cluster_data_num_L.npy"
    Sec_clu_num_path   = "./input/HNSWOPQ_with_OPQ_ver4/Second_cluster_data_num.npy"

    d   = 128
    knn = 16
    C   = 256
    HSC  = 128
    LSC  = 128

    FlatIndex = faiss.IndexFlatL2(d)
    total_top50 = np.zeros((1000000,d),dtype=int)
    total_end50 = np.zeros((1000000,d),dtype=int)
    total_top50_ID = np.zeros((1000000),dtype=int)
    total_end50_ID = np.zeros((1000000),dtype=int)

    top_num = 0
    end_num = 0
    Sec_clu_num_top = np.zeros((C,LSC),dtype=int)
    Sec_clu_num_end = np.zeros((C,HSC),dtype=int)

    for times in range (C):
        dataset   = np.load(Kmeans_path.format(str(times)))
        datasetID = np.load(Kmeans_ID_path.format(str(times)))

        dataset = np.float32(dataset)

        data_graph = np.zeros(dataset.shape[0]*knn,dtype=int)
        data_graph_index = 0
        FlatIndex.add(dataset)
        D, I = FlatIndex.search(dataset, knn)
        FlatIndex.reset()
        
        score = np.zeros(dataset.shape[0],dtype=int)

        for fir in range (I.shape[0]):
            for sec in range (knn):
                score[I[fir, sec]] += 1
        
        sort_ID = np.argsort(score)

        top_50 = dataset[sort_ID[:int(dataset.shape[0]*0.5)]]
        end_50 = dataset[sort_ID[int(dataset.shape[0]*0.5):]]
        top_50_ID = datasetID[sort_ID[:int(datasetID.shape[0]*0.5)]]
        end_50_ID = datasetID[sort_ID[int(datasetID.shape[0]*0.5):]]

        total_top50[top_num:top_num+top_50.shape[0]] = top_50
        total_end50[end_num:end_num+end_50.shape[0]] = end_50
        total_top50_ID[top_num:top_num+top_50.shape[0]] = top_50_ID
        total_end50_ID[end_num:end_num+end_50.shape[0]] = end_50_ID
        top_num += top_50.shape[0]
        end_num += end_50.shape[0]

        np.save(Sec_HNSW_path.format(str(times)),top_50)
        np.save(Sec_HNSW_ID_path.format(str(times)),top_50_ID)

        output_data, output_data_ID, H_centroids, Sec_clu_num_end[times] = Sec_kmeans(end_50, end_50_ID, HSC)
        np.save(OPQ_save_path.format(str(times)), output_data)
        np.save(OPQ_ID_save_path.format(str(times)), output_data_ID)
        np.save(OPQ_centorid_path.format(str(times)),H_centroids)

        output_data, output_data_ID, L_centroids, Sec_clu_num_top[times] = Sec_kmeans(top_50, top_50_ID, LSC)
        
        np.save(Sec_HNSW_path.format(str(times)), output_data)
        np.save(Sec_HNSW_ID_path.format(str(times)), output_data_ID)
        np.save(HNSW_centorid_path.format(str(times)),L_centroids)

        np.save(centorid_path.format(str(times)),np.concatenate((H_centroids,L_centroids)))

    np.save('./input/HNSWOPQ_with_OPQ_ver4/total_HNSW.npy',total_top50[:top_num])
    np.save('./input/HNSWOPQ_with_OPQ_ver4/total_OPQ.npy',total_end50[:end_num])
    np.save('./input/HNSWOPQ_with_OPQ_ver4/total_HNSW_ID.npy',total_top50_ID[:top_num])
    np.save('./input/HNSWOPQ_with_OPQ_ver4/total_OPQ_ID.npy',total_end50_ID[:end_num])
    np.save(Sec_clu_num_H_path,Sec_clu_num_end)
    np.save(Sec_clu_num_L_path,Sec_clu_num_top)
    np.save(Sec_clu_num_path,np.concatenate((Sec_clu_num_end,Sec_clu_num_top),axis=1))
