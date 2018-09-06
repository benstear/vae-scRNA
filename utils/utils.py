#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Data(object):

    def __init__(self,datamatrix, labels=None, test=False,
                 test_size=0.2, shuffle=True, seed=None, 
                 drop_remainder=True):

        self.datamatrix = datamatrix
        self.labels = labels
        self.test = test
        self.test_size = test_size
        self.shuffle = shuffle
        self.seed = seed
        self.train_idx = 0
        self.test_idx = 0
        self.drop_remainder = drop_remainder
        self.n_cells, self.n_dims = np.shape(self.datamatrix)

        if type(self.datamatrix) is not np.ndarray:
            self.datamatrix = np.asarray(self.datamatrix)
            
        if self.labels is None:
            self.labels = np.zeros(self.n_cells)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.datamatrix,
            self.labels,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.seed)
        
  
    def train_batch(self, batch_size):
        ''' Returns training batches of batch_size chunks '''
        
        self.batch_size = int(batch_size)
        
        if self.batch_size > self.n_cells:
            raise ValueError('Batch_size must be > number of samples')
            
        if self.train_idx + self.batch_size > len(self.x_train):
            
            if self.drop_remainder is False: 
                
                remainder = len(self.x_train)- self.train_idx 
                xtrain_rem = self.x_train[-remainder:,:] 
                ytrain_rem = self.y_train[-remainder:]
                self.train_idx = 0
                
                if self.shuffle:
                    self.x_train, self.y_train = shuffle(self.x_train,
                                                         self.y_train,
                                                        random_state=self.seed)
                return xtrain_rem, ytrain_rem
                    
            else: 
                self.train_idx = 0 
                if self.shuffle: 
                    self.x_train, self.y_train = shuffle(self.x_train,
                                                         self.y_train,
                                                        random_state=self.seed)
                
        x_train_batch = self.x_train[(self.train_idx):(self.train_idx
                                     +self.batch_size), :]
        y_train_batch = self.y_train[(self.train_idx):(self.train_idx
                                     +self.batch_size)]
        self.train_idx += batch_size
        
        return [x_train_batch, y_train_batch]


    def test_batch(self, batch_size):
        ''' Returns testing batches of batch_size chunks '''
        
        self.batch_size = int(batch_size)
        
        if self.batch_size > self.n_cells:
            raise ValueError('Batch_size must be > number of samples')

        if self.test_idx + self.batch_size > len(self.x_test):
                
            if self.drop_remainder is False:
                
                remainder = len(self.x_test)- self.test_idx
                xtest_rem = self.x_test[-remainder:,:]
                ytest_rem = self.y_test[-remainder:]
                self.test_idx = 0          
                
                if self.shuffle:
                    self.x_test, self.y_test = shuffle(self.x_test,self.y_test,
                                                    random_state=self.seed)
                return xtest_rem, ytest_rem
            
            else:
                self.test_idx = 0
                
                if self.shuffle:
                    self.x_test, self.y_test = shuffle(self.x_test,self.y_test,
                                                    random_state=self.seed)
    
        x_test_batch = self.x_test[(self.test_idx):(self.test_idx
                                   +self.batch_size), :]
        y_test_batch = self.y_test[(self.test_idx):(self.test_idx
                                   +self.batch_size)]
        self.test_idx += batch_size
        
        return [x_test_batch, y_test_batch]
