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
    collate_fn=my_collate
)

# lambda x:(
#         torch.cat(
#             [x[i][j].unsqueeze(0) for i in range(len(x))], 0
#             ).unsqueeze(0) for j in range(len(x[0]))
#         )

for i, j in loader:
    print(i)
    print(j)


for elem in loader:
    # print(len(elem[0]))
    # print(elem)
    for i, j in elem:
        print(i, j)
