import paddle

def create_toy_data(func, interval, sample_num, noise = 0.0, add_outlier = False, outlier_ratio = 0.001):
    """
    根据给定的函数，生成样本
    输入：
       - func：函数
       - interval： x的取值范围
       - sample_num： 样本数目
       - noise： 噪声均方差
       - add_outlier：是否生成异常值
       - outlier_ratio：异常值占比
    输出：
       - X: 特征数据，shape=[n_samples,1]
       - y: 标签数据，shape=[n_samples,1]
    """

    # 均匀采样
    # 使用paddle.rand在生成sample_num个随机数
    X = paddle.rand(shape = [sample_num]) * (interval[1]-interval[0]) + interval[0]
    y = func(X)

    # 生成高斯分布的标签噪声
    # 使用paddle.normal生成0均值，noise标准差的数据
    epsilon = paddle.normal(0,noise,paddle.to_tensor(y.shape[0]))
    y = y + epsilon
    if add_outlier:     # 生成额外的异常点
        outlier_num = int(len(y)*outlier_ratio)
        if outlier_num != 0:
            # 使用paddle.randint生成服从均匀分布的、范围在[0, len(y))的随机Tensor
            outlier_idx = paddle.randint(len(y),shape = [outlier_num])
            y[outlier_idx] = y[outlier_idx] * 5
    return X, y