import json
import os
from tqdm import tqdm
from pydub import AudioSegment
from utils.reader import load_audio


# 生成数据列表
def get_data_list(infodata_path, zhvoice_path):
    # 打开我们的数据文件
    with open(infodata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 训练集合测试集的文件
    f_train = open(os.path.join(zhvoice_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(zhvoice_path, 'test_list.txt'), 'w')

    sound_sum = 0
    speakers = []
    speakers_dict = {}
    # 遍历文件
    for line in tqdm(lines):
        # 加载我们的数据集，里面包括文字和音频信息
        line = json.loads(line.replace('\n', ''))
        # 如果时间小于1.3s就跳过
        duration_ms = line['duration_ms']
        if duration_ms < 1300:
            continue
        # 获取说话人信息，相当于个人编号
        speaker = line['speaker']
        # 我们给这个人设置一个编号，编号从0开始
        if speaker not in speakers:
            speakers_dict[speaker] = len(speakers)
            speakers.append(speaker)
        label = speakers_dict[speaker]
        # 获取一下音频文件路径
        sound_path = os.path.join(zhvoice_path, line['index'])
        save_path = "%s.wav" % sound_path[:-4]
        # 把mp3格式转换为wav的格式，同时删除旧的MP3文件
        if not os.path.exists(save_path):
            try:
                wav = AudioSegment.from_mp3(sound_path)
                wav.export(save_path, format="wav")
                os.remove(sound_path)
            except Exception as e:
                print('数据出错：%s, 信息：%s' % (sound_path, e))
                continue
        #  每200条数据中取一条作为测试数据
        if sound_sum % 200 == 0:
            f_test.write('%s\t%d\n' % (save_path.replace('\\', '/'), label))
        else:
            f_train.write('%s\t%d\n' % (save_path.replace('\\', '/'), label))
        sound_sum += 1

    f_test.close()
    f_train.close()


# 删除错误音频
def remove_error_audio(data_list_path):
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines1 = []
    # 我们加载一下音频。如果可以正常加载，我们就保存，否则就忽略
    for line in tqdm(lines):
        audio_path, _ = line.split('\t')
        try:
            spec_mag = load_audio(audio_path)
            lines1.append(line)
        except Exception as e:
            print(audio_path)
            print(e)
    with open(data_list_path, 'w', encoding='utf-8') as f:
        for line in lines1:
            f.write(line)


if __name__ == '__main__':
    # 首先我们获取所有的数据集
    get_data_list('data/text/infodata.json', 'data')
    remove_error_audio('data/train_list.txt')
    remove_error_audio('data/test_list.txt')
