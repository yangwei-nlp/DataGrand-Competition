"""
Description :   将预处理后的数据集变换为BiLSTM+CRF模型需要的输入格式
     Author :   Yang
       Date :   2019/8/16
"""
import codecs


f_write = codecs.open('data/dg_submit.txt', 'w', encoding='utf-8')
with codecs.open('data/dg_result.txt', 'r', encoding='utf-8') as f:
    # f = codecs.open('data/dg_result.txt', 'r', encoding='utf-8')
    # lines = f.read().split('\r\n\r\n')
    line = f.readline()
    while line:
    # for line in lines:
        if line == '':
            continue
        tokens = line.split('\r\n')
        features = []
        tags = []
        for token in tokens:
            feature_tag = token.split()
            features.append(feature_tag[0])
            tags.append(feature_tag[-1])
        samples = []
        i = 0
        while i < len(features):
            sample = []
            if tags[i] == 'O':
                sample.append(features[i])
                j = i + 1
                while j < len(features) and tags[j] == 'O':
                    sample.append(features[j])
                    j += 1
                samples.append('_'.join(sample) + '/o')
            else:
                if tags[i][0] != 'B':
                    print(tags[i][0] + ' error start')
                    j = i + 1
                else:
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/' + tags[i][-1])
            i = j
        f_write.write('  '.join(samples) + '\n')
