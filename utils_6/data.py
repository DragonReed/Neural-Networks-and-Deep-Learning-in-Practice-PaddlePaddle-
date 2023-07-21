import os
import random
import numpy as np

# 加载数据集
def load_imdb_data(path):
    assert os.path.exists(path) 
    trainset, devset, testset = [], [], []
    with open(os.path.join(path, "train.txt"), "r") as fr:
        for line in fr:
            sentence_label, sentence = line.strip().lower().split("\t", maxsplit=1)
            trainset.append((sentence, sentence_label))

    with open(os.path.join(path, "dev.txt"), "r") as fr:
        for line in fr:
            sentence_label, sentence = line.strip().lower().split("\t", maxsplit=1)
            devset.append((sentence, sentence_label))

    with open(os.path.join(path, "test.txt"), "r") as fr:
        for line in fr:
            sentence_label, sentence = line.strip().lower().split("\t", maxsplit=1)
            testset.append((sentence, sentence_label))

    return trainset, devset, testset

# 加载词典
def load_vocab(path):
    assert os.path.exists(path)
    words = []
    with open(path, "r", encoding="utf-8") as f:
        words = f.readlines()
        words = [word.strip() for word in words if word.strip()]
    word2id = dict(zip(words, range(len(words))))
    return word2id

# 将数据转换为词典id
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        sentence = sentence.split(" ")
        sentence = [word2id_dict[word] if word in word2id_dict \
                    else word2id_dict['[oov]'] for word in sentence]    
        data_set.append((sentence, sentence_label))
    return data_set

# 构造训练数据，每次传入模型一个batch，一个batch里面有batch_size条样本
def build_batch1(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle = True, drop_last = True):

    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num): 

        #每个epoch前都shuffle一下数据，有助于提高模型训练的效果。但是对于预测任务，不要做数据shuffle
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])
            
            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
    if not drop_last and len(sentence_batch) > 0:
        yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")

# 构造训练数据，每次传入模型一个batch，一个batch里面有batch_size条样本
def build_batch(word2id_dict, corpus, batch_size, max_seq_len, shuffle = True, drop_last = True):

    sentence_batch = []
    sentence_label_batch = []

    #每个epoch前都shuffle一下数据，有助于提高模型训练的效果。但是对于预测任务，不要做数据shuffle
    if shuffle:
        random.shuffle(corpus)

    for sentence, sentence_label in corpus:
        sentence_sample = sentence[:min(max_seq_len, len(sentence))]
        if len(sentence_sample) < max_seq_len:
            for _ in range(max_seq_len - len(sentence_sample)):
                sentence_sample.append(word2id_dict['[pad]'])
        
        sentence_batch.append(sentence_sample)
        sentence_label_batch.append([sentence_label])

        if len(sentence_batch) == batch_size:
            yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
            sentence_batch = []
            sentence_label_batch = []

    if not drop_last and len(sentence_batch) > 0:
        yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")

