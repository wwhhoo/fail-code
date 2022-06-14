import numpy as np
import faiss

data_128D_path      = '/home/lab/devdata/Data/128D_Mem/128D_Sec_Clu_{}_Mem.npy'
data_32D_path       = '/home/lab/devdata/Data/Sec_Mem/Sec_Clu_{}_Mem.npy'
# data_ID_path        = '/home/lab/devdata/Data/Sec_Idx/Sec_Clu_{}_MemIdx.npy'
new_data_path_PQ    = './PQ_train1M.npy'
# new_data_path_HNSW  = './'
first_stage_codebook = np.load('/home/lab/devdata/Data/FirstStageCodebook.npy')
first_stage_codebook = np.float32(first_stage_codebook)
max_num = 1000000
data_1M = np.zeros((max_num,128),dtype=int)
index = 0

for i in range (4096):
    # Load data
    data128  = np.load(data_128D_path.format(str(i)))
    data32   = np.load(data_32D_path.format(str(i)))
    data32 = np.float32(data32)
    # data_ID      = np.load(data_ID_path.format(str(i)))
    # Total data

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(data_32D_path.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(data32)

    D, I = gpu_index_flat.search(xq, k)
   

print(index)
np.save(new_data_path_PQ,data_1M)