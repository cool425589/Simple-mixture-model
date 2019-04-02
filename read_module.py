import os
import numpy as np
import re
from view import progressbar
import sys
def Get_subfile_list(filename):
    read_path = os.path.join(os.path.sep,os.getcwd(),filename)
    subfile_list = os.listdir(read_path)
    return subfile_list
def Get_read_data_func(filepath):
    def read_document_data(subfileName):
        read_path = os.path.join(os.path.sep,os.getcwd(),filepath,subfileName)
        assert os.path.isfile(read_path) , read_path + "not exists"
        with open(read_path, 'r') as f:
            return np.asarray(re.findall('(?<=\s)\d+(?=\s)|^\d+(?=\s)|(?<=\s)\d+$', f.read())).astype(np.integer)         
    return np.frompyfunc(read_document_data, 1, 1)
def init_count_matrix_func(Name_tf_dic, word_size):
    def update_count(read_data, document_number):
        counts = np.bincount(read_data,  minlength = word_size)
        with np.errstate(divide='ignore'):     
            Name_tf_dic[document_number, :] = counts
        progressbar(document_number+1, Name_tf_dic.shape[0])
    return np.frompyfunc(update_count, 2, 0)
def Get_count_matrix(FolderName, document_size, word_size):
    count_matrix = np.zeros((document_size, word_size))
    print ('now load document ')
    init_count_matrix_func(count_matrix, word_size)(Get_read_data_func(FolderName)(Get_subfile_list(FolderName)), np.arange(len(Get_subfile_list(FolderName))))
    return count_matrix
def Get_read_BGLM(BGLM_path):
    lines = 0
    BGLM_idf = []
    with open(BGLM_path) as f:
        BGLM_file = f.readlines()
        for line in BGLM_file:
            BGLM_idf.append(float(line.split()[1]))
    return np.asarray(BGLM_idf)