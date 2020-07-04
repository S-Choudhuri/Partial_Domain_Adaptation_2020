import torch.utils.data as data
from PIL import Image
import os
import sys


class DataLoader(data.Dataset):

	def __init__(self, filepath, inp_dims, transform=None):

		self.img_data, self.img_labels = self.data_loader(filepath, inp_dims)
		self.transform = transform
		self.len_data = len(self.img_labels)

	def __getitem__(self, item):

		imgs, labels = self.img_data[item], self.img_labels[item]

		if self.transform is not None:
			imgs = self.transform(imgs)
			labels = int(labels)

		return imgs, labels

	def pil_loader(path):
		# Return the RGB variant of input image
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def data_loader(self, filepath, inp_dims):
		# Load images and corresponding labels from the text file, stack them in numpy arrays and return
		if not os.path.isfile(filepath):
			print("File path {} does not exist. Exiting...".format(filepath))
			sys.exit() 
		img = []
		label = []
		with open(filepath) as fp:
			for line in fp:
				token = line.split()
				i = pil_loader(token[0])
				i = i.resize((inp_dims[0], inp_dims[1]), Image.ANTIALIAS)
				img.append(np.array(i))
				label.append(int(token[1]))
		img = np.array(img)
		label = np.array(label)
		return img, label

	def __len__(self):
		return self.len_data
