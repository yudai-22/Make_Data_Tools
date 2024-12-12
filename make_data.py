from scipy.signal import fftconvolve
import numpy as np
import torch


def parallel_processing(function, target):#並列処理
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(function, enumerate(target)), total=len(target)))
        
    return results



def slide(data, square_size):
    _, height, width = data.shape
    step = square_size // 4  # 正方形の4分の1のサイズ
    
    crops = []
    for y in range(0, height - square_size + 1, step):
        for x in range(0, width - square_size + 1, step):
            # 指定した正方形サイズの画像を切り抜く
            crop = data[:, y:y + square_size, x:x + square_size]
            crops.append(crop)
            
    return crops



def remove_nan(data_list):#データリストからnanを含むデータを除いたリストを返す
    no_nan_list = [data for data in data_list if not np.isnan(data).any()]

    return no_nan_list



def select_top(data_list, value):#valueには上位〇%の〇を入れる
    sums = np.array([np.sum(arr) for arr in data_list])
    print("平均値: ", np.mean(sums))
    #閾値計算
    parcent = 100 - value
    threshold = np.nanpercentile(sums, parcent)
    print("閾値: ", threshold)
    #上位〇%を抽出
    top_quarter_arrays = [arr for arr, s in zip(data_list, sums) if s >= threshold]

    return top_quarter_arrays



def gaussian_filter_3D(data3d):#三次元データを一層ずつガウシアンフィルター
    #ガウシアンフィルターの定義
    gaussian_num = [1, 4, 6, 4, 1]
    gaussian_filter = np.outer(gaussian_num, gaussian_num)
    gaussian_filter2 = gaussian_filter/np.sum(gaussian_filter)
    
    gau_map_list = []
    for i in range(len(data3d)):
        gau = fftconvolve(data3d[i], gaussian_filter2, mode="valid")
        gau_map_list.append(gau)
    gau_map = np.stack(gau_map_list, axis=0)

    return gau_map



def normalization(data_list):
    norm_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        norm_list.append(norm_data)
    return norm_list



def resize(data, size):
    # NumPy配列をTorchテンソルに変換
    data_torch = torch.from_numpy(data).unsqueeze(0)  # バッチ次元追加 (1, depth, height, width)
    
    # リサイズを実行 (depthを変更せず、高さと幅のみ)
    resized_data = F.interpolate(data_torch, size=size, mode="bilinear", align_corners=False)
    
    # バッチ次元を削除し、NumPy配列に戻す
    resized_data = resized_data.squeeze(0).numpy()
    
    return resized_data



def data_integrate(data):
    integ_data = np.nansum(data, axis=0)

    return integ_data



def integrate_to_x_layers(data, layers):
    """
    任意の深さを持つ三次元データをx層に積分する。
    """
    original_depth = data.shape[0]
    target_depth = layers
    
    # 元の深さをx等分するインデックスを計算
    edges = np.linspace(0, original_depth, target_depth + 1, dtype=int)
    
    # 新しい層に対する積分を計算
    integrated_layers = []
    for i in range(target_depth):
        start, end = edges[i], edges[i + 1]
        # 範囲内を積分（単純合計）
        integrated_layer = np.sum(data[start:end], axis=0)
        integrated_layers.append(integrated_layer)
    
    # x層に統一されたデータを返す
    return np.stack(integrated_layers)
