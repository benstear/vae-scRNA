#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:41:02 2018

@author: stearb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import utils
from utils import Data
# https://jmetzen.github.io/2015-11-27/vae.html
import tensorflow as tf
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
#from matplotlib.pyplot import scatter, figure, subplot, savefig
#%matplotlib inline
plt.switch_backend('agg')

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import time
#import datetime as dt

np.random.seed(0)
tf.set_random_seed(0)

################## Dataset #1 n_1078 ######
try: data
except:
    data = pd.read_csv('/Users/stearb/desktop/vae-scRNA-master/data/n_1078/data.csv')
    #data  =  pd.read_csv('/scr1/users/stearb/scdata/n_1078/data.csv')  
data = shuffle(data)
celltypes = data['TYPE'] # save cell type vector in case we need it later
y = data['Labels'] # save labels
X = data.drop(['Labels','TYPE'],axis=1) 
assert not np.any(np.isnan(X))
assert not np.any(np.isnan(y)) 
dataset = '1,078 single cells from: https://portals.broadinstitute.org/single_cell/study/atlas-of-human-blood-dendritic-cells-and-monocytes'
############################################
'''
 ################ Dataset #2 n_13,313 #####
try: data13k, labels13k
except: 
    data13k = pd.read_csv('/Users/stearb/desktop/vae-scRNA-master/data/n_13313_singlenuc/Mouse_Processed_GTEx_Data.DGE.log-UMI-Counts.txt',sep='\t')

    #data13k  =  pd.read_csv('/scr1/users/stearb/scdata/n_13313/Mouse_Processed_GTEx_Data.DGE.log-UMI-Counts.txt',sep='\t')  
    data13k = data13k.T
    labels13k  =  pd.read_csv('/scr1/users/stearb/scdata/n_13313/metadata_singleNuc_13k.txt',sep='\t')
##############################################

data13k_clipped = data13k.iloc[1:] # clip off first row (gene names)   
labels13k_clipped = labels13k.iloc[1:]  # same idea for labels
y = labels13k_clipped # save labels
X = data13k_clipped.values
assert len(labels13k_clipped) == len(data13k_clipped)
y = y['ClusterID'] 
dataset = '13,313 single cells from: https://portals.broadinstitute.org/single_cell/study/dronc-seq-single-nucleus-rna-seq-on-mouse-archived-brain#study-download'

############## Dataset #3 n_27,499 ###########
t = time.time()
data27k  =  pd.read_csv('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_27499_bipolar/exp_matrix.txt',sep='\t')  
labels27k = pd.read_csv('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_27499_bipolar/clust_retinal_bipolar.txt',sep='\t')
#time_elapsed = time.time() - t#print(time_elapsed)#print(np.shape(data27k))#print(np.shape(labels27k))
data27k = data27k.T
CLUSTERS = labels27k['CLUSTER']
y = CLUSTERS.iloc[1:]
data27k_clipped = data27k.iloc[1:]
X = data27k_clipped.values
dataset = '27,499 single cells from: '
#########################################
'''

## FOR CLUSTER 
print('DATA LOADED')

##################################
cellcount, genecount = np.shape(X)
n_samples = cellcount
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=144)

# Create Data Object for batch retrieval
Utils = Data(X,y)
Utils = utils.Data(X,y)

# ###### # ###### # ######### # ##### #  Everything above here should be excluded
###############3##################3###################################





def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.  """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=1e-7, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.logs_path = 'path'
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        ######################################
        '''
        gpu=True
        if gpu:
            # Configure for gpu capability
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True # pylint: disable=E1101
            self.sess = tf.InteractiveSession(config=config)
            self.sess.run(tf.global_variables_initializer())
        else:'''
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        
        # Set up summaries for tensorboard
        self.tf.summary.scalar('Loss',self.total_loss)
        #self.tf.summary.scalar('acc',acc)
        self.merged_summary = tf.summary.merge_all()
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(self.logs_path,grapth=tf.get_default_graph)


        #########################################
        
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latentspace
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon   REPARAMETERIZED________///
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1']),name='rec_lay1') 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2']),name='rec_lay2') 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'],name='z_mean')
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
        
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):  
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),1)
        
        # Use b-VAE technique: beta should be >1
        beta = 1.0
        
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.   set prior distro lower than -0.5 if your getting nan's
        latent_loss = beta*(-0.50 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1))
        self.total_loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
    
        # RETURN LOSSES SEPERATELY
        self.recon_loss = tf.reduce_mean(reconstr_loss)
        self.latent_loss = tf.reduce_mean(latent_loss)

        # Use ADAM optimizer
        self.optimizer= \
             tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        Return cost of mini-batch."""  # split cost
        _, total_loss, recon_loss, latent_loss, summary = self.sess.run([self.optimizer,
                                                                self.total_loss,
                                                                self.recon_loss,
                                                                self.latent_loss, self.merged_summary],
                                                               feed_dict={self.x: X})
        return total_loss, recon_loss, latent_loss
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space. """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X})
    
print('VAE model created...')




def train(network_architecture, learning_rate=1e-6,
          batch_size=200, training_epochs=5):
    
    vae = VariationalAutoencoder(network_architecture,learning_rate=learning_rate, batch_size=batch_size)
    
    cluster = True
    loss_total_vec = []
    loss_recon_vec = []
    loss_latent_vec = []
    tolerance = 0.1
    patience = 5
    patience_cntr = 0
    total_batch = int(n_samples // batch_size)
    t = time.time()
        
    # Training cycle
    for epoch in range(training_epochs): # implement early stopping
        avg_loss_total = 0.
        avg_loss_recon = 0.
        avg_loss_latent = 0.
        # Loop over all batches
        for i in range(total_batch):
      
            X_train_batch, y_train_batch = Utils.train_batch(batch_size=batch_size)
            total_loss, recon_loss, latent_loss, summary = vae.partial_fit(X_train_batch)
            summary_writer.add_summary(summary,epoch*total_batch+i)
            
            # Compute average loss's
            avg_loss_total += total_loss / n_samples * batch_size
            avg_loss_recon += recon_loss / n_samples * batch_size
            avg_loss_latent += latent_loss / n_samples * batch_size
            
        loss_total_vec = np.append(loss_total_vec,avg_loss_total)
        loss_recon_vec = np.append(loss_recon_vec,avg_loss_recon)
        loss_latent_vec = np.append(loss_latent_vec,avg_loss_latent)
        print('Epoch:', '%03d' % (epoch+1),
              "Total Loss= {:.9}, Recon Loss= {:.9}, Latent Loss= {:.9}".format(avg_loss_total,
                                                                                  avg_loss_recon,
                                                                                  avg_loss_latent))
        # Check for early stopping
        if avg_loss_total<0 or np.isnan(avg_loss_total):  # if cost goes negative or nan, stop training. 
            Utils.plot_loss(epoch,loss_total_vec,loss_recon_vec,loss_latent_vec,cluster)
            Utils.save_model(vae,cluster)
            return vae
        if loss_total_vec[epoch-1] - loss_total_vec[epoch] < tolerance: # early stopping
            patience_cntr += 1
            if patience_cntr == patience:
                Utils.plot_loss(epoch,loss_total_vec,loss_recon_vec,loss_latent_vec,cluster)
                Utils.save_model(vae,cluster)
            return vae 
     
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # no stats
    weights_stats = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) # gets stats too
    #print(tf.global_variables) # gets stats too
    # just layer1 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'layer1')
    # https://github.com/google/prettytensor/issues/6
    t_end = time.time() - t        
    Utils.plot_loss(epoch,loss_total_vec,loss_recon_vec,loss_latent_vec,cluster)
    Utils.save_params(epoch,avg_loss_recon,avg_loss_latent,learning_rate, t_end,dataset, cluster)
    #Utils.save_model(vae,cluster)
    #print(vae.z)
    #tf.Session().run(Add_3:0)  # Add_3:0
    #n_z_vals = sess.run('Add_3:0')

    return vae

print('Done Creating VAE...')


###########################################################    EVERYTHING BELOW SHOULD BE IN 
# Define architecture and train                                 A FILE CALLED TEST_vae
############################################################

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=genecount, # total feature space (# of genes)
         n_z=2)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=10) # if latent loss starts to grow, lower b, as training continue      
print('Done training VAE...')

















