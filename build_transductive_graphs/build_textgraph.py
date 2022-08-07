import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils.utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm     
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
# import sys       
from scipy.spatial.distance import cosine
# sys.path.append('/content/drive/My Drive/ColabNotebooks/My_GraphNas/graphnas')      
# sys.path.append('/content/drive/My Drive/ColabNotebooks/My_GraphNas/graphnas/utils')      
# path ='/content/drive/My Drive/ColabNotebooks/My_GraphNas/graphnas/'           
# path1 = '/content/drive/My Drive/ColabNotebooks/My_GraphNas/graphnas/'     
      
# if len(sys.argv) != 2:     
# 	sys.exit("Use: python build_textgraph.py <dataset>")
def buildgraph(dataset,path):      
    
    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr','SST-B', 'SST']  
    # dataset ='R8'                
    dim = 300                             
    
    # for dataset,dim in zip(datasets,dims):  
    # build corpus            
    # dataset = sys.argv[1]                                
    window_size = 15               
    if dataset not in datasets:          
    	sys.exit("wrong dataset name")                               
    word_vector_map = {}                         
    # Read Word Vectors                                     
    word_vector_file = path+ '/glove/glove.6B.'+str(dim)+'d.txt'
    _, embd, word_vector_map = loadWord2Vec(word_vector_file)
    word_embeddings_dim = dim    
            
    print(dataset)
    # shulffing
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []
    
    #reading document-name list 
    #doc_test_list document names for test 
    #doc_train_list document names for training
    f = open( path+ '/data_for_transd_Learn/SeqGR/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()
    # print(doc_train_list)
    # print(doc_test_list)
    # reading the content of the dataset after cleaning each doument in one line
    #doc_content_list contains all dataset
    doc_content_list = []
    f = open( path+ '/data_for_transd_Learn/SeqGR/corpus/' + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    # print(doc_content_list)
    # the ids of the training documents 
    # writing those ids into a file
    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    # print(train_ids)
    random.shuffle(train_ids)
    
    # partial labeled data
    # writting the ids of training docs after shuffling 
    train_ids_str = '\n'.join(str(index) for index in train_ids)
    f = open(  path+ '/data_for_transd_Learn/SeqGR/' + dataset + '.train.index', 'w')
    f.write(train_ids_str)
    f.close()
    # the ids of the test documents 
    # writing those ids into a file
    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)
    
    test_ids_str = '\n'.join(str(index) for index in test_ids)
    f = open( path+ '/data_for_transd_Learn/SeqGR/' + dataset + '.test.index', 'w')
    f.write(test_ids_str)
    f.close()
    #combine all ids 
    ids = train_ids + test_ids
    # print(ids)
    # print(len(ids))
    # shuffle the document name list according to the ids shuffle 
    # writting the docs names after shuffling 
    # shuffle_doc_words_list contains all dataset
    # shuffle_doc_words_list contains all docs (dataset) after shuffling 
    # writting the docs name and content after shuffling 
    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
    
    f = open( path+ '/data_for_transd_Learn/SeqGR/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_name_str)
    f.close()
    
    f = open( path+ '/data_for_transd_Learn/SeqGR/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()
    
    # build vocab
    #building the vocabulary and the frequecny of each word in the whole corpus
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    vocab = list(word_set)
    vocab_size = len(vocab)
    
    #word_doc_list a dictionary for words and the documents that appear 
    # {'of':[1,5,7]}
    word_doc_list = {}
    
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)
    
    #word_doc_freq a dictionary for words and the number of documents appears in
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)
    #word_id_map a dictionary for words and the id of each word
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
    
    # combining voabs into one string and write it into file
    vocab_str = '\n'.join(vocab)
    
    f = open( path+ '/data_for_transd_Learn/SeqGR/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()   
    
    # label list
    #label_list a  label list for all documents
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)
    
    label_list_str = '\n'.join(label_list)
    f = open( path+ '/data_for_transd_Learn/SeqGR/corpus/' + dataset + '_labels.txt', 'w')
    f.write(label_list_str)
    f.close()
    
    # x: feature vectors of training docs, no initial features
    # slect 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates
    # documents names after selecting the evaluation docs
    
    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)
    
    f = open( path+ '/data_for_transd_Learn/SeqGR/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()
    
    # building a matrix with size the number of training docs * word_embedding_dim 
    # initiall the matrix contains zeros ; no external information are used
    
    row_x = []
    col_x = []
    data_x = []
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)
    
        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len
    
    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
    #  x contains real train docs * embedding 
    # we can use external information like word2net
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))
    #print(x.shape)
    # y is the lables of the documents after converting into one-hot array
    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    # print(y)
    
    # tx: feature vectors of test docs, no initial features
    test_size = len(test_ids)
    
    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)
    
        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len
    
    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    #  x contains real train docs * embedding 
    # we can use external information like word2net
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))
    # building the one-hot array for the labels in the test set
    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)
    # print(ty)
    
    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words
    # the size is docs+vocabs * word_embeddings_dim
    # word_vectors a matrix (words * embeddings)
    # we can add external information like word2net fro each word vector
    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))
    
    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector
    
    row_allx = []
    col_allx = []
    data_allx = []
    # allx train_size (before removing the validation) * embeddings
    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)
    
        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
    #  here also add the word vectors for the data_allx which contains docs vectors 
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))
    
    
    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)
    # allx train_size (before removing the validation) * embeddings
    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))
    # print(allx.shape)
    # ally contains the labels for the training set as one-hot array
    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)
    #  here also append to the ally all words labels as one-hot array with just zeros
    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)
    
    ally = np.array(ally)
    
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
    #(2, 300) (2, 2) (2, 300) (2, 2) (464, 300) (464, 2)
    '''
    Doc word heterogeneous graph
    '''
    
    # word co-occurence with context windows         
    #creating windows of 20 words            
            
    windows = []
    
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
    #print(window)
    
    #print(shuffle_doc_words_list)
    # word_window_freq contains the frequency of the words in the windows
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
    
    #print(word_id_map)
    row = []
    col = []
    weight = []
    
    # pmi as weights
    #pmi for each pair words
    num_window = len(windows)
    
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
    #print(vocab)
    #print(train_size)
    # word vector cosine similarity as weights
    
    '''
    for i in range(vocab_size):
        for j in range(vocab_size):
            if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
                vector_i = np.array(word_vector_map[vocab[i]])
                vector_j = np.array(word_vector_map[vocab[j]])
                similarity = 1.0 - cosine(vector_i, vector_j)
                if similarity > 0.9:
                    print(vocab[i], vocab[j], similarity)
                    row.append(train_size + i)
                    col.append(train_size + j)
                    weight.append(similarity)
    '''
    # print(word_doc_freq)
    # doc word frequency
    # doc * word    the frequency of each word in specific document
    print(vocab_size)
    doc_word_freq = {}
    
    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    # building adjacency matrix with weight
    # the size is node_size * node_size
    # this contains the nodes of word and documents
    # the documents are all the documents in training or test sets
                
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)
    
    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    print(adj.shape)
    # dump objects   
    #x is a training doc * word-embedding-dim array that contains zero if noexternal information are used
    f = open( path+ "/data_for_transd_Learn/SeqGR/ind.{}.x".format(dataset), 'wb')
    
    pkl.dump(x, f)
    f.close()
    # y is the one-hot array for trainging document lables
    f = open(path+ "/data_for_transd_Learn/SeqGR/ind.{}.y".format(dataset), 'wb')
    pkl.dump(y, f)
    f.close()
    
    #tx is a test doc * word-embedding-dim array that contains zero if noexternal information are used
    f = open(path+ "/data_for_transd_Learn/SeqGR/ind.{}.tx".format(dataset), 'wb')
    pkl.dump(tx, f)
    f.close()
    
    # ty is the one-hot array for test document lables
    f = open(path+ "/data_for_transd_Learn/SeqGR/ind.{}.ty".format(dataset), 'wb')
    pkl.dump(ty, f)
    f.close()
    
    #x is a training doc + vocab * word-embedding-dim array that contains zero if no external information are used
    # the vocab to vocab uniform is used
    f = open(path+ "/data_for_transd_Learn/SeqGR/ind.{}.allx".format(dataset), 'wb')       
    pkl.dump(allx, f)
    f.close()
    #ally is a training doc+vocab * num_of_labels array that 
    print(ally.shape)
    f = open(path+ "/data_for_transd_Learn/SeqGR/ind.{}.ally".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()
    #adj is the adjacency matrix all docs+vocabs 
    f = open(path+ "/data_for_transd_Learn/SeqGR/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj, f)        
    f.close()             
