# -*- coding: utf-8 -*-

import numpy as np
from models import BiRNN_EncDec
import cPickle as pkl

def main():
    source, target = load_data()
    
    n_vocab = 30000
    dim_word = 384
    dimctx = 1024
    dim = 512
    maxlen=100
    
    x, x_mask, y, y_mask  = prepare_data(source, target, maxlen)
    
    model = BiRNN_EncDec(n_vocab, dim_word, dimctx, dim)
    model.train(x, x_mask, y, y_mask)

def prepare_data(seqs_x, seqs_y, maxlen=None):
    if maxlen != None:
        seqs_x, seqs_y = zip(*filter(lambda xy : len(xy[0]) < maxlen and len(xy[1]) < maxlen, zip(seqs_x, seqs_y)))   
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]
        if len(lengths_x) < 1:
            return None, None, None, None
    
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1
    
    x = np.zeros((maxlen_x, n_samples)).astype('int64')
    y = np.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    
    for idx, [s_x, s_y] in enumerate(zip(seqs_x,seqs_y)):
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx]+1,idx] = 1.
        y[:lengths_y[idx],idx] = s_y
        y_mask[:lengths_y[idx]+1,idx] = 1.

    return x, x_mask, y, y_mask     

def load_data():
        
    source = '/Users/seonhoon/Desktop/workspace_python/dl4mt-material-master/data/test/test_en.tok'
    source = open(source, 'r')
    target = '/Users/seonhoon/Desktop/workspace_python/dl4mt-material-master/data/test/test_fr.tok'
    target = open(target, 'r')
    source_dict = '/Users/seonhoon/Desktop/workspace_python/dl4mt-material-master/data/test/test_en.tok.pkl'
    target_dict = '/Users/seonhoon/Desktop/workspace_python/dl4mt-material-master/data/test/test_fr.tok.pkl'
    


    with open(source_dict, 'rb') as f:
        source_dict = pkl.load(f)
    with open(target_dict, 'rb') as f:
        target_dict = pkl.load(f)
    new_source=[]
    new_target=[]
    while True:

        ss = source.readline()
        if ss == "":
            break;
        ss = [source_dict[w] if w in source_dict else 1 for w in ss.split()]
        ss = [w if w < 300000 else 1 for w in ss]
        
        tt = target.readline()
        tt = [target_dict[w] if w in target_dict else 1 for w in tt.split()]
        tt = [w if w < 300000 else 1 for w in tt]
        new_source.append(ss)
        new_target.append(tt)
        
    for i in range(len(new_source)):
        if(len(new_source[i])>100):
            new_source[i]=new_source[i][:100]
        if(len(new_target[i])>100):
            new_target[i]=new_target[i][:100]
    return new_source, new_target
if __name__ == '__main__':
    main()

