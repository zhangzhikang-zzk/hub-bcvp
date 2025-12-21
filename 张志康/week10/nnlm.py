#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer, BertConfig
"""
基于pytorch的BERT语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, bert_config=None):
        super(LanguageModel, self).__init__()
        # 使用BERT配置
        if bert_config is None:
            # 默认配置适合较小的数据集
            bert_config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                max_position_embeddings=128
            )
        self.bert = BertModel(bert_config)
        self.classify = nn.Linear(bert_config.hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # 添加ignore_index参数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # BERT前向传播
        outputs = self.bert(x)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # 分类层
        y_pred = self.classify(sequence_output)  # (batch_size, seq_len, vocab_size)
        
        if y is not None:
            # 只计算非-mask位置的损失
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

# 加载字表
def build_vocab(vocab_path):
    vocab = {}
    # 首先添加特殊标记到特定的索引位置（连续索引）
    special_tokens = [
        ("<pad>", 0),
        ("<UNK>", 1), 
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("[MASK]", 4)
    ]
    for token, idx in special_tokens:
        vocab[token] = idx
    
    # 从索引5开始添加词汇表内容
    valid_index = 5
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()  # 去掉结尾换行符
            if char:  # 跳过空行
                vocab[char] = valid_index
                valid_index += 1
    # 确保最大索引是 len(vocab) - 1
    max_index = max(vocab.values())
    expected_max_index = len(vocab) - 1
    if max_index != expected_max_index:
        print(f"警告：最大索引({max_index})与词汇表大小-1({expected_max_index})不匹配，正在调整...")
        # 重新映射索引以确保连续性
        new_vocab = {}
        sorted_items = sorted(vocab.items(), key=lambda x: x[1])
        for i, (word, old_idx) in enumerate(sorted_items):
            new_vocab[word] = i
        return new_vocab
    return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    encodings = ["utf-8", "gbk"]
    for encoding in encodings:
        try:
            with open(path, encoding=encoding) as f:
                for line in f:
                    corpus += line.strip()
            break
        except UnicodeDecodeError:
            continue
    return corpus

# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    
    # 将输入转换为带有[CLS]和[SEP]标记的形式
    x = [vocab["[CLS]"]]  # 开始标记
    x_words = [vocab.get(word, vocab["<UNK>"]) for word in window]
    x.extend(x_words)
    x.append(vocab["[SEP]"])  # 结束标记
    
    # 构建目标序列，对应位置的字符
    y = []
    y.append(-100)  # [CLS]标记不需要预测
    y_words = [vocab.get(word, vocab["<UNK>"]) for word in target]
    y.extend(y_words)
    y.append(-100)  # [SEP]标记不需要预测
    
    # 应用BERT的mask机制
    x, y = apply_bert_mask(x, y, vocab)
    
    return x, y

# 实现BERT的mask机制
def apply_bert_mask(x, y, vocab, mask_prob=0.15):
    x = x.copy()
    y_masked = [-100] * len(y)  # 初始化为忽略的标签
    
    # 对除了[CLS]和[SEP]之外的所有位置应用mask
    for i in range(1, len(x) - 1):  # 不包括[CLS]和[SEP]
        rand = random.random()
        if rand < mask_prob:
            # 80%的概率替换为[MASK]
            if rand < mask_prob * 0.8:
                x[i] = vocab["[MASK]"]
            # 10%的概率替换为随机词
            elif rand < mask_prob * 0.9:
                x[i] = random.randint(5, len(vocab) - 2)  # 随机词（排除特殊标记和最后一个索引）
            # 10%的概率保持不变（使用原词）
            # 注意：y_masked[i]只有在进行mask操作时才设置为实际标签
            y_masked[i] = y[i]
        else:
            # 不mask的位置不参与损失计算
            y_masked[i] = -100
    
    return x, y_masked

# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    max_vocab_index = len(vocab) - 1
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        # 调试检查：确保所有索引都在有效范围内
        max_x = max(x)
        if max_x > max_vocab_index:
            raise ValueError(f"发现超出词汇表范围的索引：{max_x} (最大允许值：{max_vocab_index})")
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab_size):
    model = LanguageModel(vocab_size)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # 构造输入序列，包含[CLS]和[SEP]
            x = [vocab.get("[CLS]", vocab["<UNK>"])]
            words = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x.extend(words)
            x.append(vocab.get("[SEP]", vocab["<UNK>"]))
            
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-2]  # 取倒数第二个位置的结果（最后一个有效字符）
            index = sampling_strategy(y)
            pred_char = reverse_vocab.get(index, "<UNK>")
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            # 构造输入序列
            x = [vocab.get("[CLS]", vocab["<UNK>"])]
            words = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x.extend(words)
            x.append(vocab.get("[SEP]", vocab["<UNK>"]))
            
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-2]  # 取倒数第二个位置的结果
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        # 训练轮数
    batch_size = 32       # 每次训练样本个数（由于BERT对序列长度敏感，减小batch size）
    train_sample = 10000   # 每轮训练总共训练的样本总数（减少样本数以便快速训练）
    window_size = 32       # 样本文本长度（增加窗口大小以适应更多上下文）
    vocab = build_vocab("vocab.txt")       # 建立字表
    corpus = load_corpus(corpus_path)     # 加载语料
    model = build_model(len(vocab))    # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   # 降低学习率以稳定训练
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)