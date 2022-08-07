import argparse
import numpy as np
import codecs
import pickle
import time
import zipfile
from math import log  

def buildgraph(dataset,path ):
    # experiment setting      
    pretrained = True
    d_pretrained = 300
    seed = 1111
    path_data = path +'/data_for_induct_Learn/SeqGR/'
    path_glove = path +'/glove/'          
    np.random.seed(seed)                     
    if dataset not in ['R8', 'R52', 'ohsumed', 'mr','SST-2','SST-1']:
        raise ValueError('Data {data} not supported, currently supports "R8", "R52",  "R52new","SST-2","SST-1 and ohsumed".')               

    # read files     
    print('\n[info] Dataset:', dataset)           
    time_start = time.time()           

    label2idx = read_label(path_data  + '/'+ dataset +'.txt')               
    word2idx = build_vocab(path_data  + '/corpus/'+ dataset +'.txt')      
    print('word number:', len(word2idx))           
    n_class = len(label2idx)
    args =  {'pretrained':pretrained,'d_pretrained':d_pretrained, 'seed':seed,'n_class':n_class}
    n_word = len(word2idx)
    print('\tTotal classes:', n_class)
    print('\tTotal words:', n_word)

    embeds = get_embedding(d_pretrained,pretrained,path_glove, word2idx)
    All_content = read_allcontent(path_data  + '/'+ dataset +'.txt', path_data + '/corpus/'+ dataset +'.txt')     
    weight     = pmi(All_content, word2idx,list(word2idx.keys()),len(All_content))       
    tr_data, tr_gt,val_data,val_gt,te_data,te_gt, tr_text, val_text , te_text = read_corpus( path_data  + '/corpus/'+ dataset +'.txt',path_data  + '/'+ dataset +'.txt', label2idx, word2idx)
    
    train_mask = [1] * len(tr_text)
    val_mask = [0] * len(tr_text)
    test_mask = [0] * len(tr_text)
    
    train_mask.extend([0] * len(val_text)) 
    val_mask.extend([1] * len(val_text)) 
    test_mask.extend( [0] * len(val_text))
    
    train_mask.extend([0] * len(te_text)) 
    val_mask.extend([0] * len(te_text))          
    test_mask.extend( [1] * len(te_text))          
    # save processed data          
    mappings = {
        'label2idx': label2idx,
        'word2idx': word2idx,
        'tr_data': tr_data,
        'tr_gt': tr_gt,
        'val_data': val_data,
        'val_gt': val_gt,
        'te_data': te_data,
        'te_gt': te_gt,
        'embeds': embeds,
        'weight': weight,
        'args': args,              
        'tr_text': tr_text, 
        'val_text':val_text , 
        'te_text':te_text,
        'train_mask' :train_mask,
        'val_mask' :val_mask,
        'test_mask' : test_mask,
        'All_content': All_content
    }                                                          
    with open(path_data + dataset + '.pkl', 'wb') as f:
        pickle.dump(mappings, f)                                           

    print('\n[info] Time consumed: {:.2f}s'.format(time.time() - time_start))    
     
def read_label(path):
    """ Extract and encode labels. """
    with open(path) as f:
        labels = [line.split('\t')[2] for line in f.read().split('\n')]
    labels = list(set(labels))      
    return {label: i for i, label in enumerate(labels)}       

def build_vocab(corpus_path):      
    
    with open(corpus_path) as f:
        content = [line for line in f.read().split('\n')]
     
    word_set = set()
    i = 0
    for doc_words in content:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            
    word2idx = {word: i + 1 for i, word in enumerate(word_set)}
    word2idx['<pad>'] = 0

    return word2idx                  

def read_allcontent(labels, corpus):             
    
    with open(corpus) as f:     
        content = [line for line in f.read().split('\n')]    

    return content
    
def vocab_per(train_path):       
    
    with open(train_path) as f:
        content = [line.split('\t')[1] for line in f.read().split('\n')]
    word_set = set()      
    i = 0
    for doc_words in content:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            
    word2idx = {word: i + 1 for i, word in enumerate(word_set)}       
    word2idx['<pad>'] = 0     
    return word2idx         

def read_vocab(path):
    """ Extract words from vocab and encode. """
    with open(path) as f:
        words = f.read().split('\n')
    word2idx = {word: i + 1 for i, word in enumerate(words)}
    word2idx['<pad>'] = 0

    return word2idx

def read_corpus(corpus_path,label_path, label2idx, word2idx):
    """ Encode both corpus and labels. """      
    tr_val_data = []    
    tr_val_gt  = []
    tr_data = []    
    tr_gt  = []
    val_data=[]
    val_gt  =[]
    te_data= []
    te_gt = []
    tr_val_text =[]
    tr_text   = []
    val_text = []
    te_text   = []      
    
    with open(corpus_path) as f:
        content = [line for line in f.read().split('\n')]
    with open(label_path) as f:
        labels = [line.split('\t') for line in f.read().split('\n')]
    for i, (text,label ) in enumerate(zip(content, labels )):                    
        if label[1] == 'train':
            tr_val_data.append([encode_word(word, word2idx) for word in text.split()])    
            tr_val_gt.append(label2idx[label[2]])
            tr_val_text.append(text)
        else:
            te_data.append([encode_word(word, word2idx) for word in text.split()])    
            te_gt.append(label2idx[label[2]])
            te_text.append(text)      
    train_size = int(len(tr_val_data) * 0.9)            
    val_size  = len(tr_val_data)  - train_size
    tr_data = tr_val_data[:train_size]     
    tr_text = tr_val_text[:train_size]
    val_text = tr_val_text[train_size:]
    tr_gt  = tr_val_gt[:train_size]
    val_data=tr_val_data[train_size:]    
    val_gt  =tr_val_gt[train_size:]     
    return tr_data, tr_gt,val_data,val_gt,te_data,te_gt, tr_text, val_text , te_text

def read_content(path):           
    """ Encode both corpus and labels. """
    with open(path) as f:
        content = [line.split('\t')[1] for line in f.read().split('\n')]
    return content

def encode_word(word, word2idx):
    """ Encode word considering unknown word. """
    try:
        idx = word2idx[word]
    except KeyError:
        idx = word2idx['UNK']
    return idx    

def get_embedding(d_pretrained,pretrained,path_glove,  word2idx):             
    """ Find words in pretrained GloVe embeddings. """          
    if pretrained:         
        path = path_glove + 'glove.6B.' + str(d_pretrained) + 'd.txt'
        embeds_word = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word2idx), d_pretrained))
        emb_counts = 0
        for i, line in enumerate(codecs.open(path, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == (d_pretrained + 1) and s[0] in word2idx:
                embeds_word[word2idx[s[0]]] = np.array([float(i) for i in s[1:]])
                emb_counts += 1

        embeds_word[0] = np.zeros_like(embeds_word[0])  # <pad>

    else:
        embeds_word = None
        emb_counts = 'disabled pretrained'

    print('\tPretrained GloVe found:', emb_counts)
    return embeds_word

def pmi(docs_list, word_id_map,vocab,train_size):
    #creating windows of 20 words            
    window_size = 15        
    windows = []
    
    for doc_words in docs_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:   
                word_window_freq[window[i]] = 1
            appeared.add(window[i])        
    #word_pair_count the coocurrence of the pair words in the windows; the key is the ids of the two words
    # the freq will be stored in two orders for example the words 0,100 and 100,0       
    word_pair_count = {}   
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    row = []
    col = []
    weight = {}
    
    #pmi for each pair words
    num_window = len(windows)     
    print('word_pair_count: ', len(word_pair_count))                        
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])     
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i-1]]   
        word_freq_j = word_window_freq[vocab[j-1]]    
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)      
        col.append(train_size + j)        
        weight[key]= pmi
    print('len weight : ', len(weight))     
    return weight    
# if __name__ == '__main__':
#     main()
