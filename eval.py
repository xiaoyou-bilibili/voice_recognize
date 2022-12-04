import numpy as np
import torch
from tqdm import tqdm

from utils.reader import load_audio

# 评估的一些参数
list_path = 'data/test_list.txt' # 测试集地址
input_shape = (1, 257, 257) # 音频输入形状
model_path = 'model/resnet34.pth' # 模型路径
device = torch.device("cuda") # 设备类型

# 加载模型
model = torch.jit.load(model_path)
model.to(device)
model.eval()


# 根据对角余弦值计算准确率
def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_accuracy = 0
    best_threshold = 0
    for i in tqdm(range(0, 100)):
        threshold = i * 0.01
        y_test = (y_score >= threshold)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_accuracy, best_threshold


# 预测音频
def infer(audio_path):
    data = load_audio(audio_path, mode='test', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    feature = model(data)
    return feature.data.cpu().numpy()[0]


def get_all_audio_feature(list_path):
    with open(list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    features, labels = [], []
    print('开始提取全部的音频特征...')
    for line in tqdm(lines):
        path, label = line.replace('\n', '').split('\t')
        feature = infer(path)
        features.append(feature)
        labels.append(int(label))
    return features, labels


# 计算对角余弦值
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def main():
    # 计算所有音频的特征
    features, labels = get_all_audio_feature(list_path)
    scores = []
    y_true = []
    print('开始两两对比音频特征...')
    for i in tqdm(range(len(features))):
        feature_1 = features[i]
        for j in range(i, len(features)):
            feature_2 = features[j]
            score = cosin_metric(feature_1, feature_2)
            scores.append(score)
            y_true.append(int(labels[i] == labels[j]))
    accuracy, threshold = cal_accuracy(scores, y_true)
    print('当阈值为%f, 准确率最大，为：%f' % (threshold, accuracy))


if __name__ == '__main__':
    main()
