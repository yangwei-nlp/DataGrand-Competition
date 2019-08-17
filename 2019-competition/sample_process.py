"""
Description :   将原始数据集预处理，最终效果:每一行一个字符和对应标注，不同句子之间隔一行
     Author :   Yang
       Date :   2019/8/17
"""
import codecs
import os


# 处理训练集
with codecs.open('data/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    results = []
    for line in lines:
        features = []
        tags = []
        samples = line.strip().split('  ')
        for sample in samples:
            sample_list = sample[:-2].split('_')
            tag = sample[-1]
            features.extend(sample_list)
            tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(['B-' + tag] + ['I-' + tag] * (len(sample_list)-1))
        results.append(dict({'features': features, 'tags': tags}))
    train_write_list = []
    with codecs.open('data/dg_train.txt', 'w', encoding='utf-8') as f_out:
        for result in results:
            for i in range(len(result['tags'])):
                train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
            train_write_list.append('\n')
        f_out.writelines(train_write_list)

# step 2 test data in
with codecs.open('data/test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    results = []
    for line in lines:
        features = []
        sample_list = line.split('_')
        features.extend(sample_list)
        results.append(dict({'features': features}))
    test_write_list = []
    with codecs.open('data/dg_test.txt', 'w', encoding='utf-8') as f_out:
        for result in results:
            for i in range(len(result['features'])):
                test_write_list.append(result['features'][i] + '\n')
            test_write_list.append('\n')
        f_out.writelines(test_write_list)
