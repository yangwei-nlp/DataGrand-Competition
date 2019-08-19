"""
Description :   
     Author :   Yang
       Date :   2019/8/15
"""
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(1)

import os
os.chdir("D:\\..courses\\DataGrand-Competition\\2019-competition")


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 构造模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 5
        self.hidden_dim = hidden_dim  # 4
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # 5

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 隐藏层的初始化参数
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()  # 参数的格式(tensor shape(2,1,2), tensor shape(2,1,2))
        # print(self.word_embeds(sentence).shape)  # 词嵌入后得到矩阵shape,(11, 5)
        test = self.word_embeds(sentence)  # 报错
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # print(embeds.shape)  # (11, 1, 5)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # nn.LSTM的背后原理有待研究
        # print(lstm_out.shape, len(self.hidden))  # (11,1,4)和2
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)  # 可以视为lstm对句子每个单词的标注
        # print(lstm_feats.shape)  # (11,5)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # sentence: (11,)
        print(sentence)
        lstm_feats = self._get_lstm_features(sentence)  # (11,5)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class Data(Dataset):
    def __init__(self, Xtrain, ytrain, word_to_ix, tag_to_ix):
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.Xtrain = Xtrain
        self.ytrain = ytrain

    def __len__(self):
        return len(self.Xtrain)

    def __getitem__(self, idx):
        sentence = prepare_sequence(self.Xtrain[idx], self.word_to_ix)
        tags = torch.tensor([self.tag_to_ix[t] for t in self.ytrain[idx]], dtype=torch.long)
        return tags, sentence


# 准备数据
EMBEDDING_DIM = 128
HIDDEN_DIM = 64

STOP_TAG = "<STOP>"
START_TAG = "<START>"

X = []  # 所有样本，每个元素代表一个样本(一句话)的特征
y = []  # 每个元素代表一个样本的所有的标注
word_to_ix = {'PADDING': 0}  # 词语-索引映射
tag_to_ix = {STOP_TAG: 0, START_TAG: 1}  # 标注集
with codecs.open('data/dg_train.txt', 'r', encoding='utf-8') as f:
    # f = codecs.open('data/dg_train.txt', 'r', encoding='utf-8')
    line = f.readline()
    features = []  # 每句话的所有词
    tags = []  # 每句话中每个词的标注
    while line:
        if line == '\n':
            X.append(features)
            y.append(tags)
            features = []
            tags = []
        else:
            word = line.split('\n')[0].split('\t')[0]
            tag = line.split('\n')[0].split('\t')[1]
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
            features.append(word)
            tags.append(tag)
        line = f.readline()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)


def pad_tensor(vec, pad):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to

    return:
        a new tensor padded to 'pad'
    """
    return torch.cat([vec, torch.zeros(pad - len(vec), dtype=torch.int64)], dim=0).data.numpy()


# def pad_y(vec, pad):
#     return torch.cat([vec, torch.zeros(pad - len(vec), dtype=torch.int64)], dim=0).data.numpy()


def my_collate(batch):
    xs = [torch.tensor(v[0]) for v in batch]
    ys = [torch.tensor(v[1]) for v in batch]
    # 获得每个样本的序列长度
    seq_lengths = torch.tensor([v for v in map(len, xs)])
    max_len = max([len(v) for v in xs])
    # 每个样本都padding到当前batch的最大长度
    xs = torch.tensor([pad_tensor(v, max_len) for v in xs])
    ys = torch.tensor([pad_tensor(v, max_len) for v in ys])
    # 把xs和ys按照序列长度从大到小排序
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    xs = xs[perm_idx]
    ys = ys[perm_idx]
    return xs, seq_lengths, ys


def MyDataLoader(Xtrain, ytrain, word_to_ix, tag_to_ix, batch_size):
    dataset = Data(Xtrain, ytrain, word_to_ix, tag_to_ix)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)


data_loader = MyDataLoader(Xtrain, ytrain, word_to_ix, tag_to_ix, 200)




model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(1):
    print("epoch :{}".format(epoch + 1))
    # for i, (sentence, tags) in enumerate(sample_data):
    # for i, (tags, sentence) in enumerate(data_loader):
    for i, elem in enumerate(data_loader):
        print(elem)
        # print(len(elem))

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        # if i % 1000 == 0:
        #     print("第 {} 个样本".format(i))
        # model.zero_grad()
        #
        # # Step 2. Run our forward pass.
        # loss = model.neg_log_likelihood(sentence, tags)
        #
        # # Step 3. Compute the loss, gradients, and update the parameters by
        # # calling optimizer.step()
        # loss.backward()
        # optimizer.step()
