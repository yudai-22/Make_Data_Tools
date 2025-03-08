import numpy as np
from npy_append_array import NpyAppendArray

import psutil
from tqdm.notebook import tqdm


def proccess_memmap(file_name, *data):
    data = list(data)
    shape = data[0][0].shape
    num_arrays = len(data[0])
  
    # メモリマップファイルを作成
    memmap_file = file_name + ".npy"
    print(f"file_name: {memmap_file}")
    merged_data = np.memmap(memmap_file, dtype=np.float32, mode="w+", shape=(num_arrays * len(data), *shape))
    
    # 各データをメモリマップにコピー
    for i in tqdm(range(len(data))):
        if i == len(data):
            break
        else:
            merged_data[i*num_arrays:(i+1)*num_arrays] = np.array(data[i], dtype=np.float32)
    
    # メモリから解放
    del merged_data
    print("The save has completed!!")


def proccess_npyAppend(file_name, *data):
    all_data = []
    for part in data: 
      all_data += part

    shape = all_data[0].shape
    num_arrays = len(all_data)

    mem = psutil.virtual_memory()
    total = mem.total
    used = mem.used
    free = mem.available
    
    print(f"総メモリ: {total / 2**30:.2f} GB")
    print(f"使用中のメモリ: {used / 2**30:.2f} GB")
    print(f"使用可能なメモリ: {free / 2**30:.2f} GB")
    
    data_byte = np.prod(shape) * 8
    print(f"\n一つあたりのデータ容量: {data_byte/ 2**20:.2f} MB")
    print(f"総データ容量: {(data_byte * num_arrays) / 2**30:.2f} GB")
    
    batch_size = (free // data_byte) // 3
    print(f"\n総データ数: {len(all_data)}")
    print(f"バッチサイズ:\n {batch_size}")

    file_name = file_name + ".npy"
    saved_file = NpyAppendArray(file_name)
    
    for i in tqdm(range(0, len(all_data), batch_size)):
      batch_data = np.asarray(all_data[i:i + batch_size])
      batch_data = np.ascontiguousarray(batch_data)
      saved_file.append(batch_data)

    saved_file.close()
    print("The save has completed!!")
  
