import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_embedding(param, inp):
   pass

def build_classifier(param, embedding):
   pass

def build_discriminator(param, embedding):
   pass

def build_combined_classifier(inp, classifier):

def build_combined_discriminator(inp, discriminator):
   pass

def build_combined_model(inp, comb):
   pass

class Discriminator(nn.Module):
	def __init__(self, input_dim, out_dim):

		super(Discriminator, self).__init__()

		#define the structure
		self.dom_classifier = nn.Sequential()

	def forward(self, x):

		output = self.dom_classifier(x)

		return output

class Classifier(nn.Module)
	def __init__(self, input_dim, out_dim):

		super(Classifier, self).__init__()

		#define the structure
		self.classifier = nn.Sequential()

	def forward(self, x):

		output = self.classifier(x)

		return output

class Model(nn.Module):

	def __init__(self, input_dim):

		super(Model, self).__init__()

		#private source encoder

		#private target encoder

		#shared encoder

		#shared decoder

	def forward(self, x):

		pass

    
