import os
import re
import shutil
import time
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.reader import CustomDataset
from utils.arcmargin import ArcNet
from utils.resnet import resnet34

# 训练模型的相关参数
gpus = '0' # 训练使用的GPU序号，使用英文逗号,隔开，如：0,1'
batch_size = 32 # 训练的批量大小
num_workers = 4 # 读取数据的线程数量
num_epoch = 50 # 训练的轮数
num_classes = 3242 # 分类的类别数量
learning_rate =  1e-3 #初始学习率的大小
weight_decay = 5e-4 # weight_decay的大小
lr_step = 10 # 学习率衰减步数
input_shape = (1, 257, 257) # 数据输入的形状
train_list_path = 'data/train_list.txt' # 训练数据的数据列表路径
test_list_path = 'data/test_list.txt' # 测试数据的数据列表路径
save_model = 'models/' # 模型保存的路径
resume = None # 恢复训练，当为None则不使用恢复模型
pretrained_model = None # 预训练模型的路径，当为None则不使用预训练模型

# 评估模型
@torch.no_grad()
def test(model, metric_fc, test_loader, device):
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.to(device).long()
        feature = model(spec_mag)
        output = metric_fc(feature, label)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc.item())
    return float(sum(accuracies) / len(accuracies))


# 保存模型
def save_model(model, metric_fc, optimizer, epoch_id):
    model_params_path = os.path.join(save_model, 'epoch_%d' % epoch_id)
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
    # 保存模型参数和优化方法参数
    torch.save(model.state_dict(), os.path.join(model_params_path, 'model_params.pth'))
    torch.save(metric_fc.state_dict(), os.path.join(model_params_path, 'metric_fc_params.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_params_path, 'optimizer.pth'))
    # 删除旧的模型
    old_model_path = os.path.join(save_model, 'epoch_%d' % (epoch_id - 3))
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)
    # 保存整个模型和参数
    all_model_path = os.path.join(save_model, 'resnet34.pth')
    if not os.path.exists(os.path.dirname(all_model_path)):
        os.makedirs(os.path.dirname(all_model_path))
    torch.jit.save(torch.jit.script(model), all_model_path)


def train():
    device_ids = [int(i) for i in gpus.split(',')]
    # 获取数据集
    train_dataset = CustomDataset(train_list_path, model='train', spec_len=input_shape[2])
    # 加载我们的数据集
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size * len(device_ids),
                              shuffle=True,
                              num_workers=num_workers)
    # 这边是加载我们的测试数据集
    test_dataset = CustomDataset(test_list_path, model='test', spec_len=input_shape[2])
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers)

    device = torch.device("cuda")
    # 构建restNet模型（残差结构网络），这个网络可以可以图像分类
    model = resnet34()
    # ArcFace Loss函数
    metric_fc = ArcNet(512, num_classes)

    # 如果有多个GPU，那么就多个GPU一起训练
    if len(gpus.split(',')) > 1:
        model = DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        metric_fc = DataParallel(metric_fc, device_ids=device_ids, output_device=device_ids[0])

    # 首先加载我们的模型，然后打印一下模型的结构
    model.to(device)
    metric_fc.to(device)
    if len(gpus.split(',')) > 1:
        summary(model.module, input_shape)
    else:
        summary(model, input_shape)

    # 初始化epoch数
    last_epoch = 0
    # 获取优化方法
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # 获取学习率衰减函数
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1, verbose=True)

    # 获取损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 加载模型参数和优化方法参数
    # 这里是加载一下预训练模型
    if resume:
        optimizer_state = torch.load(os.path.join(resume, 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state)
        # 获取预训练的epoch数
        last_epoch = int(re.findall('(\d+)', resume)[-1])
        if len(device_ids) > 1:
            model.module.load_state_dict(torch.load(os.path.join(resume, 'model_params.pth')))
            metric_fc.module.load_state_dict(torch.load(os.path.join(resume, 'metric_fc_params.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(resume, 'model_params.pth')))
            metric_fc.load_state_dict(torch.load(os.path.join(resume, 'metric_fc_params.pth')))
        print('成功加载模型参数和优化方法参数')

    # 开始训练
    sum_batch = len(train_loader) * (num_epoch - last_epoch)
    for epoch_id in range(last_epoch, num_epoch):
        for batch_id, data in enumerate(train_loader):
            start = time.time()
            # 获取我们的输入和标签
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            # 先调用restNet获取特征值
            feature = model(data_input)
            # 然后根据标签和特征值计算输出
            output = metric_fc(feature, label)
            # 计算loss
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每迭代100尺就打印一下准确率和loss信息
            if batch_id % 100 == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                eta_sec = ((time.time() - start) * 1000) * (sum_batch - (epoch_id - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f, lr: %f, eta: %s' % (
                    datetime.now(), epoch_id, batch_id, len(train_loader), loss.item(), acc.item(), scheduler.get_lr()[0], eta_str))
        scheduler.step()
        # 迭代完一轮后就对我们的模型进行评估
        # 开始评估
        model.eval()
        print('='*70)
        accuracy = test(model, metric_fc, test_loader, device)
        model.train()
        print('[{}] Test epoch {} Accuracy {:.5}'.format(datetime.now(), epoch_id, accuracy))
        print('='*70)

        # 保存模型
        if len(device_ids) > 1:
            save_model(model.module, metric_fc.module, optimizer, epoch_id)
        else:
            save_model(model, metric_fc, optimizer, epoch_id)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    train()
