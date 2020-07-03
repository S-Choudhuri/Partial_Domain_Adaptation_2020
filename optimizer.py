import numpy as np
import torch.optim as optim

def opt_classifier(model, param):
  
  config = {'lr': param['lr_classifier'], 'betas': (param["b1_classifier"], param["b2_classifier"])}
  opt = optim.Adam(model.parameters(), **config)
  
  return opt

def opt_discriminator(model, param):
  
  config = {'lr': param['lr_discriminator'], 'betas': (param["b1_discriminator"], param["b2_discriminator"])}
  opt = optim.Adam(model.parameters(), **config)
  
  return opt

def opt_combined(model, param):
  
  config = {'lr': param['lr_combined'], 'betas': (param["b1_combined"], param["b2_combined"])}
  opt = optim.Adam(model.parameters(), **config)
  
  return opt
