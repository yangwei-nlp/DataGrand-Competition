#coding:utf-8
# seq2seq bilstm+cnn+crf
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras_contrib.layers import CRF
import argparse


class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape  

def attention_3d_block(inputs, max_len):
    a0 = NonMasking()(inputs)
    a1 = Permute((2, 1))(a0)
    a2 = Dense(max_len, activation='softmax')(a1)
    a_probs = Permute((2, 1), name='attention_vec')(a2)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')

    return output_attention_mul

class TextBLCC():
    def __init__(self, label_len, doc_max_len=450, emb_size=768):
        self.doc_max_len = doc_max_len
        self.emb_size = emb_size
        self.label_len = label_len
        self.model = self.__build_neuro_network()

    def __build_neuro_network(self): #(450,768)
        word_input_1 = Input(shape=(self.doc_max_len, self.emb_size), name='input_1')
        masked_seqs_1 = Masking(mask_value=0.)(word_input_1)
        bilstm_1 = Bidirectional(LSTM(256, activation='tanh', dropout=0.25,
                                      return_sequences=True), merge_mode='concat')(masked_seqs_1)
        # bilstm_2 = Bidirectional(LSTM(256, activation='tanh', dropout=0.25,
        #                               return_sequences=True), merge_mode='concat')(bilstm_1)

        conv_1 = Conv1D(filters=512, activation='relu',
                        kernel_size=7, padding='same')(word_input_1)
        # conv_2 = Conv1D(filters=512, activation='relu',
        #                 kernel_size=5, padding='same')(word_input_1)

        conv_d_1 = BatchNormalization()(conv_1)
        # conv_d_2 = BatchNormalization()(conv_2)
    
        # attention_mul = attention_3d_block(bilstm_1, self.doc_max_len)

    	# merge
        rnn_cnn_merge = Concatenate(axis=2)([bilstm_1, conv_d_1])
        rnn_cnn_merge = SpatialDropout1D(0.5)(rnn_cnn_merge)

        dense = TimeDistributed(Dense(self.label_len))(rnn_cnn_merge)

    	# crf
        crf = CRF(self.label_len, sparse_target=True)
        crf_output=crf(dense)

    	# build model
        model = Model(inputs=[word_input_1], outputs=[crf_output])
        model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
        model.summary()
        return model

    def train(self, x_train, y_train, batch_size=16, epochs=1000, verbose=1, validation_split=0, validation_data=None, cbs=[]):
        print('x_train shape:',x_train.shape,'y_train shape:',y_train.shape)
        self.model.fit([x_train], [y_train], batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                       validation_data=validation_data, verbose=verbose, callbacks=cbs)

def train_bert_model():
    from vizcallback import VizCallback
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=32,help="Batch_size")
    parser.add_argument("--epochs",type=int,default=5000,help="Epochs")
    parser.add_argument("--validation_split",type=float,default=0,help="Validation_split")
    FLAGS, unparsed = parser.parse_known_args()
    labels = []
    if not os.path.exists('../data/checkpoints-bert'):
        os.mkdir('../data/checkpoints-bert')
    with open('../data/train_data_s/class_indice.txt') as f:
        for line in f:
            labels.append(line.strip())
    x_train_1 = np.load('../data/train_data_s/x_data_bert.npy')

    y_train = np.load('../data/train_data_s/y_data_bert.npy')

    x_val_1 = np.load('../data/test_data_s/x_data_bert.npy')
    y_val = np.load('../data/test_data_s/y_data_bert.npy')
    print(len(labels),labels)
    model = TextBLCC(label_len=len(labels))
    
    # model.model.load_weights('../data/checkpoints-bert/weights.860-0.69.hdf5')
    cbs=[ModelCheckpoint('../data/checkpoints-bert/weights.{epoch:04d}-{val_acc:.2f}.hdf5',
            monitor='val_acc', mode='max', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir='../data/logs-bert'),
        VizCallback(test_x_ori=x_val_1, test_x=x_val_1, test_y=y_val, model_path='../data/checkpoints-bert/')]

    model.model.fit([x_train_1], [y_train], batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, # initial_epoch=860,
                    shuffle=True, validation_data=([x_val_1], [y_val]), verbose=1, callbacks=cbs)


def predict_bert_model():
    from vizcallback import VizCallback
    labels = []
    if not os.path.exists('../data/checkpoints-bert'):
        os.mkdir('../data/checkpoints-bert')
    with open('../data/train_data_s/class_indice.txt') as f:
        for line in f:
            labels.append(line.strip())
    test_vecs = np.load('../data/test_data_s/x_data_bert.npy')
    # y_val = np.load('../data/test_data_s/y_data_bert.npy')
    print(len(labels),labels)
    bert_model = TextBLCC(label_len=len(labels))
    bert_model.model.load_weights('../data/checkpoints-bert/real_acc_weights-0221-0.8525.h5')
    predict_test = (np.argmax(bert_model.model.predict([test_vecs]), axis=2)).tolist()

    pdf_test = pd.read_csv('../data/test_data_s.csv')
    pdf_test_rep = pd.DataFrame(pdf_test['doc_line'].str.replace(' ', ''))
    test_txt_list = pdf_test_rep['doc_line'].values.astype(str).tolist()
    test_docid_list = pdf_test['docid'].values.astype(str).tolist()
    test_clz_list = pdf_test['class'].values.astype(str).tolist()
    test_lno_list = pdf_test['line_no'].values.astype(str).tolist()
    acc_count = {k: 0 for k in set(test_docid_list)}
    total_count = len(test_txt_list)
    doc_line_count = {k: 0 for k in set(test_docid_list)}
    begin = time.time()
    for i, test_doc in enumerate(test_txt_list):
        test_clz = test_clz_list[i]
        test_id = test_docid_list[i]
        test_lno = test_lno_list[i]
        pred_clz = labels[predict_test[int(test_id)-1][int(test_lno)]]
        doc_line_count[test_id] += 1
        if test_clz == pred_clz:
            acc_count[test_id] += 1
        print(i, '\t', test_clz, test_doc, '\n\t', pred_clz,  'docid:', test_docid_list[i], 'line_no:', test_lno)
        print('=' * 99)
        if (i < total_count - 2 and test_id != test_docid_list[i + 1]) or i == total_count - 1:
            print('docid:', test_id, 'lines:',
                  doc_line_count[test_id], 'acc_count:', acc_count[test_id], 'acc:',
                  acc_count[test_id] / doc_line_count[test_id])
            print('end spend:', time.time() - begin, '=' * 99, '\r\n')
    print('spend:', time.time() - begin)


if __name__ == '__main__':
    train_bert_model()
    predict_bert_model()

   
    
