import os
import numpy as np
import torch

from utils.reader import load_audio

# 服务的一些参数
# 置信度
threshold = 0.6
input_shape = (1, 257, 257)
# 数据库文件位置
audio_db_path = 'audio_db'
# 特征信息和人民信息
person_feature = []
person_name = []

# model = None
# 设置为GPU启动
device = torch.device("cuda")
# 加载我们的模型
model = torch.jit.load('models/resnet34.pth')
model.to(device)
model.eval()

def infer(audio_path):
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    feature = model(data)
    return feature.data.cpu().numpy()

# 加载音频库信息
audios = os.listdir(audio_db_path)
for audio in audios:
    path = os.path.join(audio_db_path, audio)
    name = audio[:-4]
    feature = infer(path)[0]
    person_name.append(name)
    person_feature.append(feature)
    print("Loaded %s audio." % name)

# 识别我们的声音模型
def recognition(path):
    name = ''
    pro = 0
    # 返回结果
    response = []
    # 首先我们获取一下这个音频的特征信息
    feature = infer(path)[0]
    # 遍历我们存储的所有音频数据库
    for i, person_f in enumerate(person_feature):
        # 计算一下两个音频的相似度
        score = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        response.append({
            "name": person_name[i],
            "score": float(score),
            "is_origin": True if score > threshold else False
        })
    # 对列表从大到小进行排序
    response = sorted(response,key = lambda e:e['score'],reverse = True)
    return response


# 声纹注册
def register(save_path, user_name):
    feature = infer(save_path)[0]
    person_name.append(user_name)
    person_feature.append(feature)

