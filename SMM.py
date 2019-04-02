import os
import read_module as readM
import utils
import numpy as np
#from SMM_module import SMM_modules
from SMM_module import SMM_modules
from VSM import Get_rank_result
TOP_k = 2
EM_times = 10
alpha = 0.8
_lambda = 0.8
batch_size = 8
BGLM_path = 'Homework4/BGLM.txt'
query_path = 'Homework4/Query'
document_path = 'Homework4/Document'
result_path = 'Homework4/result.txt'
or_path = 'Homework4/or_result.txt'
BGLM = readM.Get_read_BGLM(BGLM_path)
word_size  = len(BGLM)
document_list = readM.Get_subfile_list(document_path)
query_list = readM.Get_subfile_list(query_path)
document_size = len(document_list)
query_size = len(query_list)
document_count = readM.Get_count_matrix(document_path, document_size, word_size)
log_document_count = utils.log_domain(document_count)
query_count = readM.Get_count_matrix(query_path, query_size, word_size)
with open(or_path, "w") as or_writefile:
    with open(result_path, "w") as writefile:
        writefile.write("Query,RetrievedDocuments")
        or_writefile.write("Query,RetrievedDocuments")

        SMM = SMM_modules(alpha, _lambda, log_document_count, BGLM, TOP_k, word_size, query_size, document_list, query_list, query_count, writefile , batch_size, EM_times, Get_rank_result())
        SMM.batch_progress()
        """
        SMM = SMM_modules(alpha, _lambda, log_document_count, BGLM, TOP_k, word_size, query_size, document_list, query_list, query_count, writefile , EM_times, or_writefile)
        SMM.batch_progress()
        """





