import numpy as np
import read_module as readM
import os
import math
import utils 
class SMM_modules:
    def __init__(self, alpha, _lambda, log_count_matrix, log_BG_matix, Relevent_document_number, word_size, query_size, document_list, query_list, query_count, writefile, batch_size, EMtimes, VSM_query_rel):
        self.alpha = alpha
        self.log_BG_matix = log_BG_matix
        self.log_Tsmm = np.zeros((batch_size, word_size))
        self.log_count_matrix = log_count_matrix  
        self.Relevent_document_number = Relevent_document_number
        self.word_size = word_size
        self.query_size = query_size
        self.Rel_count_query_matrix  = np.zeros((Relevent_document_number, word_size, batch_size))
        self.Likelihood = np.zeros((batch_size,))
        self._lambda = _lambda
        self.document_list = document_list
        self.write_file = writefile
        self.batch_size = batch_size
        self.now = 0
        self.query_count = query_count
        self.log_Psmm_matix = np.zeros((batch_size, word_size))
        self.query_list = query_list
        self.EMtimes = EMtimes
        self.writefile = writefile
        self.VSM_query_rel = VSM_query_rel
    def Update_query(self):
        return (1-self.alpha)*self.query_count[self.now:(self.now+self.batch_size),:]/((np.sum(self.query_count[self.now:(self.now+self.batch_size),:],axis = 1))[:,np.newaxis]) + (self.alpha)*utils.exp_domain(self.log_Psmm_matix)
    def Get_KL_divergence(slef):
        document_count = utils.exp_domain(slef.log_count_matrix)
        return -(np.sum( slef.Update_query().T[np.newaxis,:,:] * utils.log_domain((1-slef.alpha)*document_count/(np.sum(document_count,axis = 1)[:,np.newaxis]) + (slef.alpha)*np.exp(slef.log_BG_matix))[:,:,np.newaxis] ,axis = 1))
    def Get_rank(self,wirte_bool):
        query_list = self.query_list[self.now:(self.now+self.batch_size)]        
        KL_divergence = self.Get_KL_divergence()
        if wirte_bool == True :    
            print ('now rank query are ', query_list)
            rankresult = np.argsort(KL_divergence, axis = 0)
            for query_index in range(rankresult.shape[1]):     
                self.writefile.write("\n"+query_list[query_index]+",")  
                for file_index in rankresult[:,query_index][0:100]:
                    self.writefile.write(self.document_list[file_index]+" ")
        else :
            rankresult = np.argsort(pro_matrix, axis = 0)          
        return rankresult
    def check_normalization(self, normalization_matrix):
        num = 0
        for check in np.exp(normalization_matrix).sum(axis = 1):
            assert math.isclose(1, check, rel_tol=1e-1), " normalization_matrix shape : "+ str(normalization_matrix.shape[0] ) +","+str(normalization_matrix.shape[1] ) + "num : " + str(num) + " check : "+str(check)
            num+=1
    def check_likelihood(self): 
        new_likelihood = np.sum(np.sum(np.logaddexp(np.log(1 - self._lambda)+self.log_Psmm_matix, np.log(self._lambda)+self.log_BG_matix).T*np.exp(self.Rel_count_query_matrix), axis = 1),axis = 0)
        num = 0
        for old_like in self.Likelihood:
            info =  "New_likelihood be smaller!  L = ",  old_like, " , new_L = " , new_likelihood[num]
            assert old_like <= new_likelihood[num], info
            num+=1       
        self.Likelihood[:] = new_likelihood
    def E_step(self):
        self.log_Tsmm[:] = np.log(1 - self.alpha) + self.log_Psmm_matix - np.logaddexp( (np.log(1 - self.alpha) + self.log_Psmm_matix), (np.log(self.alpha) + self.log_BG_matix ) )
    def M_step(self):
        up = np.logaddexp.reduce ( self.log_Tsmm.T + self.Rel_count_query_matrix, axis = 0 )
        self.log_Psmm_matix[:] = (up - np.logaddexp.reduce(up, axis = 0)).T
    def Estimate(self, times):
        for step in range(self.EMtimes):
            print ('now-em ', step)
            self.E_step()
            self.M_step()
            self.check_normalization(self.log_Psmm_matix)
            self.check_likelihood()
    def renew_init_Psmm(self):
        self.log_Psmm_matix[:] = np.log((self.query_count/np.sum(self.query_count,axis = 1)[:,np.newaxis])[self.now:self.now+self.batch_size,:])
    def renew_random_Psmm(self):
        self.log_Psmm_matix[:] = utils.log_domain(utils.init_norm(np.random.randint(500, size= (self.batch_size,self.word_size))) )
    def first_Get_pseudo_Rel_feedback(self):
        for query_num in range(self.batch_size) :
            self.Rel_count_query_matrix[:, : , query_num] = self.log_count_matrix[(self.VSM_query_rel[self.now:self.now+self.batch_size,0:self.Relevent_document_number])[query_num, :], :]
    def batch_progress(self,):
        while True :
            if self.now + self.batch_size <= self.query_size:
                print ('\nnow first get pseudo feedback', str(self.now))
                self.Likelihood.fill(-1E10)
                self.renew_random_Psmm()             
                self.check_normalization(self.log_Psmm_matix)               
                self.first_Get_pseudo_Rel_feedback()
                print ('\nnow Estimate by EM')
                self.Estimate(self.EMtimes)
                self.Get_rank(True)
                self.now = self.now + self.batch_size
            else :
                break

        
        

            
    