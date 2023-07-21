import os
import numpy as np
import jieba
import matplotlib.pyplot as plt
import math

# 读取词汇表
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            word = line.strip("\n")
            vocab[word] = idx
    return vocab

def load_dataset(data_path, is_test):
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if is_test:
                    text = line.strip()
                    examples.append((text,))
                else:
                    label, text = line.strip().split('\t')
                    # label = self.label_map[label]
                    examples.append((text, label))
        return examples

def convert(text,vocab,stop_words):
    # 对文本使用结巴分词，并过滤掉停用词
    list_words=[]
    for word in jieba.cut(text):
        if word in vocab and word not in stop_words:
            list_words.append(word)
        elif word in vocab and word in stop_words:
            continue
        else:
            list_words.append(word)
    return list_words

# 加载数据集
def load_imdb_data(path):
    assert os.path.exists(path) 
    trainset, devset, testset = [], [], []
    with open(os.path.join(path, "train.txt"), "r") as fr:
        for line in fr:
            sentence_label, sentence = line.strip().lower().split("\t", maxsplit=1)
            trainset.append((sentence, int(sentence_label)))

    with open(os.path.join(path, "dev.txt"), "r") as fr:
        for line in fr:
            sentence_label, sentence = line.strip().lower().split("\t", maxsplit=1)
            devset.append((sentence, int(sentence_label)))

    with open(os.path.join(path, "test.txt"), "r") as fr:
        for line in fr:
            sentence_label, sentence = line.strip().lower().split("\t", maxsplit=1)
            testset.append((sentence, int(sentence_label)))

    return trainset, devset, testset


def load_thucnews_data(path):
    train_path=os.path.join(path,'train.txt')
    dev_path=os.path.join(path,'val.txt')
    test_path=os.path.join(path,'test.txt')

    train_data = load_dataset(train_path, False)
    dev_data = load_dataset(dev_path, False)
    test_data = load_dataset(test_path, False)
    return train_data,dev_data,test_data

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot(train_losses, val_losses, train_epoch, eval_steps=500):
    plt.figure()
    avg_train_losses = []
    steps_per_epoch = math.ceil(len(train_losses) / train_epoch)
    smooth_loss=smooth_curve(train_losses, factor=0.9)
    train_list_length = list(range(len(smooth_loss)))
    # print(train_list_length)
    plt.plot(train_list_length, smooth_loss, color='red', label="Train losses")
    val_list_length = [i*eval_steps for i in range(len(val_losses))]
    # print(val_list_length)
    plt.plot(val_list_length, val_losses, color='blue', label="Valid losses")
    plt.ylabel("losses")
    plt.legend(loc='upper left')
    plt.show()
