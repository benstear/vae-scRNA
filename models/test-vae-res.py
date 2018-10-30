#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:13:44 2018

@author: stearb
"""
from vae_res import VariationalAutoencoder,train
from utils import Data
import matplotlib.pyplot as plt
plt.switch_backend('agg')

n_1078=0
n_13313=1
n_27499=0

if n_1078:
    genecount=26594
elif n_13313:
    genecount=17308
elif n_27499:
    genecount=13166


###########################################################    
# Define architecture and train                                 
############################################################

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=genecount, # total feature space (# of genes)
         n_z=64)  # dimensionality of latent space

# Create instant of VAE
vae = VariationalAutoencoder(network_architecture,learning_rate=1e-5,batch_size=72)

# Train VAE
trained_vae = train(network_architecture, training_epochs=120, beta=0.65) # if latent loss starts to grow, lower b, as training continue      
print('Done training VAE with 0 errors...\n')
'''
X_test_batch, y_test_batch = Data.test_batch(100)
z_mu = vae.transform(X_test_batch)
plt.figure(figsize=(16, 12)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], cmap=y_test_batch)
#plt.colorbar()
plt.set(xlabel='latent dim 1', ylabel=' latent dim 2',title='VAE 2-D latent space scatter plot')

plt.grid()
cluster=False
if cluster:
    plt.savefig('/scr1/users/stearb/results/vae/vae-scatter-1078-n_z.png')
else:
    #plt.savefig('/Users/dawnstear/desktop/Mid_Atlantic_Poster/vae-logs/vae-scatter.png')
    plt.savefig('/Users/stearb/desktop/vae-scatter-1078-n_z.png')
'''





'''
# VAE are very finicky, 1. random sampling and back propogation
# genetic datasets..KL

# VAE's dont work well when n << d because the wide variance causes the 
# KL divergence to blow up. For this reason we might want to widen the 
# prior distro?, easiest way to do this is to lower the 0.5 in the KL divergence
# definition.
# Try normalizing the input to zero mean / unit variance
# Lets use some data where n << d
# what does lowering the 0.5 in KLD have to do with the b-vae?
# Try normalizing data
# and try digitizing data


# get weights for signature (test weights...after transform)
# do disease - control signature

# see if one latent z node can distinguish sex, another can distinguish mal/ben tumor, etc,    
# use yuanchao one to see if it clusters better////get 1mil dset split up//to batching function for it


# pie chart of cell type amounts---see how well it does
# finish 29744 dim red timings

# FIGs
    1. architecture
    2. main loss plot of training with 6-8 latent space
    3. dim red techniques compared to VAE train time
     xxx 4. beta loss plot
    5. cluster

# need testing func for validation loss
    
- 3 subplots for dim red times? 
- benchmark loss plot - call total loss training and training, validation 
- pie charts for data?
- cluster? 


ignore = 
test, 3 epoch = 3261492

'''

