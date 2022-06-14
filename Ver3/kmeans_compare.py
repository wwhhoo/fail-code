import numpy as np
import faiss


if __name__ == '__main__':

    Kmeans_path        = './input/SIFT1M_Kmeans256/SIFT1M_Kmeans256_{}.npy'
    Kmeans_ID_path     = './input/SIFT1M_Kmeans256_ID/SIFT1M_Kmeans256_ID_{}.npy'
    Sec_Kmeans_path    = './input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256/Sec_Kmeans{}_{}.npy'
    Sec_Kmeans_ID_path = './input/HNSWOPQ_with_OPQ_ver3/Sec_Kmeans256_ID/Sec_Kmeans{}_{}_ID.npy'
    centorid_path      = "./input/HNSWOPQ_with_OPQ_ver3/centroids/kmeans_centroids_{}.npy"


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


        end_50 = dataset
        end_50_ID = datasetID

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
        output_data = np.zeros((SC,20000,d),dtype=np.uint8)
        data_ID     = np.zeros((SC,20000),dtype=np.uint32)
        end_50      = np.uint8(end_50)

        for i in range(end_50.shape[0]):
   
            output_data[I[i],count[I[i]]] = end_50[i]
            data_ID[I[i],count[I[i]]]     = end_50_ID[i]
            count[I[i]] += 1

        for i in range(output_data.shape[0]):
            np.save(Sec_Kmeans_path.format(str(times),str(i)), output_data[i,:count[i]])
            np.save(Sec_Kmeans_ID_path.format(str(times),str(i)), data_ID[i,:count[i]])
        np.save(centorid_path.format(str(times)),kmeans.centroids)
