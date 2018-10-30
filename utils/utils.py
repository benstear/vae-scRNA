#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd
from matplotlib.pyplot import scatter, figure, subplot, savefig
import matplotlib.pyplot as plt
import os
import datetime as dt


class Data(object):

    def __init__(self,datamatrix, labels=None,test_size=0.2, shuffle=True, 
                 seed=None, drop_remainder=True):

        self.datamatrix = datamatrix
        self.labels = labels
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
            
        if np.size(self.datamatrix,0) == np.size(self.labels,0) is False:
            raise ValueError('Data and Labels must be the same length')


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
                return [xtrain_rem, ytrain_rem]
                    
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
                return [xtest_rem, ytest_rem]
            
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
    
    
    def save_model(self, model,cluster=False):
        
        if cluster:
            log_dir = '/scr1/users/stearb/results/vaeplot_'
        else:
            log_dir= '/Users/dawnstear/desktop/Mid_Atlantic_Poster/vae-logs/'
        app_name='vae_model_'+dt.datetime.now().strftime("%m:%d--%H-%M-")
        saver = tf.train.Saver() # to find size (bytes) of model, leave empty , {"n_z": n_z}
        saver.save(self.sess, os.path.join(log_dir,app_name))
        
        return
    
    def plot_loss(self,epoch,loss_total_vec,loss_recon_vec,loss_latent_vec,beta, dataset, cluster=True):    # add in unique color for beta value
        
        if cluster: 
            log_dir = '/scr1/users/stearb/results/vae/plots/vaeplot'+'-%s-' % dataset+'_beta_%.3f' % beta +'_'
        else:
            log_dir = '/Users/dawnstear/desktop/Mid_Atlantic_Poster/vae-logs/vaeplot_'
        
        fig, ax = plt.subplots(figsize=(20, 10))
        epochs = range(epoch+1)
        
        ax.plot(epochs,loss_total_vec)
        ax.plot(epochs,loss_recon_vec)
        ax.plot(epochs,loss_latent_vec)
        
        ax.legend(['Total Loss','Reconstruction Loss',
                   'KL Divergence (Latent) Loss'])
    
        # ax.set(xlabel='Epoch', ylabel=' Loss',title='Losses Over Training Phase, beta = %f' % beta)  
        ax.set(xlabel='Epochs', ylabel='Reconstruction Loss',title='VAE Average Loss per Epoch')  # dont include beta in title
        #ax.xlim([1, epoch+2])
        ax.grid()
        fig.savefig(log_dir+dt.datetime.now().strftime("%m-%d--%H-%M")+'.png')
        return
    
    def save_params(self,epoch,avg_loss_recon,avg_loss_latent,
                        loss_latent_vec,loss_recon_vec,learning_rate,
                        t_end,dataset,beta,loss_total_vec, network_architecture, cluster=True):
        
        if cluster: log_dir = '/scr1/users/stearb/results/vae/parameters/'
        else: log_dir = '/Users/dawnstear/desktop/Mid_Atlantic_Poster/vae-logs/vae_parameters_'
        f = open(log_dir+'%s-' % dataset+'_beta_%.3f-' % beta + dt.datetime.now().strftime("%m-%d--%H:%M") +'.txt','w')
        f.write('\t\t\t\t -------------TRAINING STATS-------------\n\n\n')
        f.write('Dataset Used:\t %s\n' % dataset)
        f.write('\nEpochs Completed: %s\n\n' % (epoch+1))
        f.write('\nLatent Space Dimensions, z = %d\n\n' % network_architecture['n_z'] )
        #f.write('Network architecture: %')
        f.write('Learning Rate:\t%s\n' % learning_rate)
        f.write('Beta Value:\t%s\n\n' % beta)
        f.write('FINAL LOSSES-   Reconstruction Loss: %s \t Latent Loss: %s\t\n\n' % (avg_loss_recon,avg_loss_latent))
        f.write('Reconstruction loss = %s\n\n\n' % loss_recon_vec)
        f.write('KL divergence loss = %s\n\n\n' % loss_latent_vec)
        f.write('Total Loss = %s\n\n\n' % loss_total_vec)
        f.write('Training Time: %s minutes and %s seconds\n\n\n' % (t_end//60,t_end%60))
        f.close()
        return
    
        #def drt_params(params):
        # write to a single text file
        #tsne_params = tsne.get_params()
        #params, use dict.
    #def save_architecture(tf.get_default_graph):
    
    # Combine 



