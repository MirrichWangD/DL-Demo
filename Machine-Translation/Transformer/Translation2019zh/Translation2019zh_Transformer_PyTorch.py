# -*- coding: UTF-8 -*-
"""
+++++++++++++++++++++++++++++++++++
@ File        : Translation2019zh_Transformer_PyTorch.py
@ Time        : 2023/4/25 9:19
@ Author      : Mirrich Wang
@ Version     : Python 3.8.12 (Conda)
+++++++++++++++++++++++++++++++++++
原链接：https://mp.weixin.qq.com/s/PyYH0EK1cLiXvBsOb5xmSQ
数据集：https://github.com/brightmart/nlp_chinese_corpus Translation2019zh 英译中数据集
+++++++++++++++++++++++++++++++++++
"""

# 导入基础模块
from pathlib import Path
from collections import Counter
import json
import time
import os

# 导入运算模块
import numpy as np
import torch
from torch.nn import Transformer
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 导入分词模块
import jieba

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
SPECIALS = ["<sos>", "<eos>", "<unk>", "<pad>"]
SOS_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3

"""+++++++++++++++++++
@@@ 数据集预处理
+++++++++++++++++++"""


def read_file(json_path):
    english_sentences = []
    chinese_sentences = []
    tokenizer = get_tokenizer('basic_english')
    with open(json_path, 'r', encoding="utf-8") as fp:
        for line in fp:
            line = json.loads(line)
            english, chinese = line['english'], line['chinese']
            # Correct mislabeled data
            if not english.isascii():
                english, chinese = chinese, english
            # Tokenize
            english = tokenizer(english)
            chinese = list(jieba.cut(chinese))
            chinese = [x for x in chinese if x not in {' ', '\t'}]
            english_sentences.append(english)
            chinese_sentences.append(chinese)
    return english_sentences, chinese_sentences


def create_vocab(sentences, max_element=None):
    """Note that max_element includes special characters"""

    default_list = ['<sos>', '<eos>', '<unk>', '<pad>']

    char_set = Counter()
    for sentence in sentences:
        c_set = Counter(sentence)
        char_set.update(c_set)

    if max_element is None:
        return default_list + list(char_set.keys())
    else:
        max_element -= 4
        words_freq = char_set.most_common(max_element)
        # pair array to double array
        words, freq = zip(*words_freq)
        return default_list + list(words)


def sentence_to_tensor(sentences, vocab):
    vocab_map = {k: i for i, k in enumerate(vocab)}

    def process_word(word):
        return vocab_map.get(word, UNK_ID)

    res = []
    for sentence in sentences:
        sentence = np.array(list(map(process_word, sentence)), dtype=np.int32)
        res.append(sentence)

    return np.array(res, dtype=object)


def tensor_to_sentence(tensor, mapping, insert_space=False):
    res = ''
    first_word = True
    for id in tensor:
        word = mapping[int(id.item())]

        if insert_space and not first_word:
            res += ' '
        first_word = False

        res += word

    return res


def main():
    en_sens, zh_sens = read_file(
        'data/translation2019zh_valid.json')
    print(*en_sens[0:3])
    print(*zh_sens[0:3])
    en_vocab = create_vocab(en_sens, 10000)
    zh_vocab = create_vocab(zh_sens, 30000)
    print(list(en_vocab)[0:10])
    print(list(zh_vocab)[0:10])
    # np.save('data/translation2019zh/en_vocab.npy', en_vocab)
    # np.save('data/translation2019zh/zh_vocab.npy', zh_vocab)

    # en_vocab = np.load('data/translation2019zh/en_dict.npy')
    # zh_vocab = np.load('data/translation2019zh/zh_dict.npy')

    en_tensors = sentence_to_tensor(en_sens, en_vocab)
    zh_tensors = sentence_to_tensor(zh_sens, zh_vocab)

    print(tensor_to_sentence(en_tensors[0], en_vocab, True))
    print(tensor_to_sentence(zh_tensors[0], zh_vocab))

    # np.save('data/translation2019zh/en_sentences.npy', en_tensors)
    # np.save('data/translation2019zh/zh_sentences.npy', zh_tensors)

    # en_tensors = np.load('data/translation2019zh/en_sentences.npy',
    #                      allow_pickle=True)
    # zh_tensors = np.load('data/translation2019zh/zh_sentences.npy',
    #                      allow_pickle=True)


if __name__ == '__main__':
    main()
