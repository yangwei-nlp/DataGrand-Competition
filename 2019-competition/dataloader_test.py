"""
Description :   
     Author :   Yang
       Date :   2019/8/18
"""
import torch
import torch.utils.data as Data
import numpy as np

test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

inputing = torch.tensor(np.array([test[i:i + 3] for i in range(10)]))
target = torch.tensor(np.array([test[i:i + 1] for i in range(10)]))

torch_dataset = Data.TensorDataset(inputing, target)
batch = 3

def my_collate(batch):
    return [[x.numpy().tolist(), y.numpy().tolist()] for x, y in batch]


loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch,  # 批大小
    # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
    # collate_fn=lambda x: x
    collate_fn=lambda x: x
)

def pad_tensor(vec, pad):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to

    return:
        a new tensor padded to 'pad'
    """
    return torch.cat([vec, torch.zeros(pad - len(vec), dtype=torch.float)], dim=0).data.numpy()


def my_collate(batch):
    xs = [torch.tensor(v[0]) for v in batch]
    ys = torch.tensor([v[1] for v in batch])
    # 获得每个样本的序列长度
    seq_lengths = torch.tensor([v for v in map(len, xs)])
    max_len = max([len(v) for v in xs])
    # 每个样本都padding到当前batch的最大长度
    xs = torch.tensor([pad_tensor(v, max_len) for v in xs])
    # 把xs和ys按照序列长度从大到小排序
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    xs = xs[perm_idx]
    ys = ys[perm_idx]
    return xs, seq_lengths, ys

for i, j in loader:
    print(i)
    print(j)


for elem in loader:
    # print(len(elem))
    print(elem)
    # print(type(elem))
    # for i, j in elem:
    #     print(i, j)
