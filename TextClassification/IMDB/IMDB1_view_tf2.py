# coding: utf-8

# 导入所需的模块
import os
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 查看当前工作目录
print(os.getcwd())

""" 
# 自动导入IMDB数据集
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data() 
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
train_text = x_train
y_train = y_train
test_text = x_test
y_test = y_test
"""

# 手动导入数据集
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "./data/IMDB/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print("downloaded:", result)

if not os.path.exists("./data/IMDB/aclImdb"):
    tfile = tarfile.open("./data/IMDB/aclImdb_v1.tar.gz", "r:gz")
    result = tfile.extractall("IMDB/")

# 使用正则表达式删除HTML的标签
import re


def rm_tags(text):
    re_tag = re.compile(r"<[^>]+>")
    return re_tag.sub("", text)


# 使用read_files()函数读取IMDB文件目录
def read_files(filetype):
    path = "./data/IMDB/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print("read", filetype, "files:", len(file_list))

    all_labels = [1] * 12500 + [0] * 12500

    all_texts = []
    for fi in file_list:
        with open(fi, encoding="utf8") as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts


# 导入的数据
y_train, train_text = read_files("train")
y_test, test_text = read_files("test")

# 查看数据，0-12499项：正面评价文字，0-12499项：正面评价，全是“1”。
# 12500-24999：负面评价文字，12500-24999：负面评价,全是”0”
len(train_text)
len(test_text)
train_text[0]
y_train[0]

train_text[12501]
y_train[12501]

# 1.x版本与2.0版本对照
# from keras.preprocessing import sequence
# tf.keras.preprocessing.sequence
# from tf.keras.preprocessing.text import Tokenizer
# tf.keras.preprocessing.text.Tokenizer()


# 建立token，即用训练的25000评价文字产生一个字典，
# 只取排序后的前2000名英文单词进入字典（也可以取更大的数进入字典）
token = tf.keras.preprocessing.text.Tokenizer(num_words=2000)
token.fit_on_texts(train_text)

# 查看是否读取了25000个评论
print(token.document_count)

# 查看每个单词在所有评论中出现的次数，
print(token.word_index)

# 使用token字典，将“影评文字”转为“数字列表”
# 将每一篇文章的文字转换一连串的数字，只有在字典中的文字才会被转换为数字
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

# 查看每个评论有多少个字，平均字长，画出分布图
avg_len = list(map(len, x_train_seq))
np.mean(avg_len)

# 直方图可视化
plt.hist(avg_len, bins=range(min(avg_len), max(avg_len) + 50, 50))
plt.show()

# 查看“数字列表”
print(train_text[0])
print(x_train_seq[0])

len(train_text[0])
len(x_train_seq[0])

len(train_text[11])
len(x_train_seq[11])

# 让转换后的数字长度相同
# 文章內的文字，转换为数字后，每一篇的文章所产生的数字长度都不同，因为之后需要进行类神经网络的训练，所以每一篇文章所产生的数字长度必须相同
# 以下列程序代码为例maxlen=100，所以每一篇文章转换为数字都必须为100
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=100)

# 如果文章转成数字大于100，会剪去“数字列表”前面的数字
print("before pad_sequences length=", len(x_train_seq[0]))
print(x_train_seq[0])
print("after pad_sequences length=", len(x_train[0]))
print(x_train[0])

# 如果文章转成数字不足100,pad_sequences处理后，前面会加上0
print("before pad_sequences length=", len(x_train_seq[1]))
print(x_train_seq[11])
print("after pad_sequences length=", len(x_train[1]))
print(x_train[11])
