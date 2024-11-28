def slide(image, square_size):
    
    _, height, width = image.shape
    step = square_size // 4  # 正方形の4分の1のサイズ
    
    crops = []
    for y in range(0, height - square_size + 1, step):
        for x in range(0, width - square_size + 1, step):
            # 指定した正方形サイズの画像を切り抜く
            crop = image[:, y:y + square_size, x:x + square_size]
            crops.append(crop)
            
    return crops


def remove_nan(data_list):
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


def data_integrate(data):
    integ_data = np.nansum(data, axis=0)

    return integ_data
