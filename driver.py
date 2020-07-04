SEED = 7
import os
import sys
import argparse
import random
import numpy as np
from tensorflow import set_random_seed
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from optimizer import *
from data_loader import *
from loss_weights import *

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
set_random_seed(SEED)
random.seed(SEED)

from PIL import Image
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

def pil_loader(path):
    # Return the RGB variant of input image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def one_hot_encoding(param):
    # Read the source and target labels from param
    s_label = param["source_label"]
    t_label = param["target_label"]

    # Encode the labels into one-hot format
    classes = (np.concatenate((s_label, t_label), axis = 0))
    num_classes = np.max(classes)
    if 0 in classes:
            num_classes = num_classes+1
    s_label = to_categorical(s_label, num_classes = num_classes)
    t_label = to_categorical(t_label, num_classes = num_classes)
    return s_label, t_label
            
def data_loader(filepath, inp_dims):
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

def batch_generator(data, batch_size):
    #Generate batches of data.
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size = batch_size, replace = False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr
        
def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print 'learning rate is set to %f' % current_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer

def train(param):
    
    #define dataloaders for source and target
	dataset_source = DataLoader(param['source_path'], param['inp_dims'])
	dataloader_source = torch.utils.data.DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=4)

	dataset_target = DataLoader(param['target_path'], param['inp_dims'])
	dataloader_target = torch.utils.data.DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=4)

	len_dataloader = min(len(dataloader_source), len(dataloader_target))
    
    #######################
	# define loss         #
	#######################
	#target/source shared and target/source private
	loss_diff = DiffLoss()

	#classifier loss
	loss_classification = torch.nn.CrossEntropyLoss()

	#reconstruction losses
	loss_recon1 = MSE()
	loss_recon2 = SIMSE()

	#discriminator adversarial loss
	loss_adv = WLoss()

	#Define models
	classifier = Classifier()

	discriminator = Discriminator()

	combined_model = Model()

	#####################
	# setup optimizer   #
	#####################
	opt_model = opt_combined(combined_model, param)
	opt_discriminator = opt_discriminator(discriminator, param)
	opt_classifier = opt_classifier(classifier, param)

    for epoch in range(param['num_iterations']):
        data_source_iter = iter(dataloader_source)
    	data_target_iter = iter(dataloader_target)
        
        i = 0
        
        while i < len_dataloader:
            
            #target training
    		data_target = data_target_iter.next()
    		t_img, t_label = data_target

    		combined_model.zero_grad()
    		classifier.zero_grad()
    		discriminator.zero_grad()

    		loss = 0
    		batch_size = len(t_label)

    		input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        	class_label = torch.LongTensor(batch_size)
        	domain_label = torch.ones(batch_size)
        	domain_label = domain_label.long()

        	input_img.resize_as_(t_img).copy_(t_img)
        	class_label.resize_as_(t_label).copy_(t_label)
        	target_inputv_img = Variable(input_img)
        	#target_classv_label = Variable(class_label)
        	target_domainv_label = Variable(domain_label)

        	target_private, target_shared, target_rec = combined_model(target_inputv_img, mode='target')
        	target_domain_label = discriminator(target_shared)

        	#target loss calculation
        	target_domain = loss_adv(target_domain_label, target_domainv_label)
        	loss += target_domain

        	target_diff = loss_diff(target_private, target_shared)
        	loss += target_diff

        	target_mse = loss_recon1(target_rec, target_inputv_img)
        	loss += target_mse

        	target_simse = loss_recon2(target_rec, target_inputv_img)
        	loss += target_simse

        	loss.backward()
        	opt_model.step()
        	opt_discriminator.step()


    		#source training
    		data_source = data_source_iter.next()
        	s_img, s_label = data_source

        	combined_model.zero_grad()
    		classifier.zero_grad()
    		discriminator.zero_grad()

    		loss = 0
    		batch_size = len(s_label)

    		input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        	class_label = torch.LongTensor(batch_size)
        	domain_label = torch.zeros(batch_size)
        	domain_label = domain_label.long()

    		input_img.resize_as_(input_img).copy_(s_img)
        	class_label.resize_as_(s_label).copy_(s_label)
        	source_inputv_img = Variable(input_img)
        	source_classv_label = Variable(class_label)
        	source_domainv_label = Variable(domain_label)

        	source_private, source_shared, source_rec = combined_model(source_inputv_img, mode='source')
        	source_domain_label = discriminator(source_shared)
        	source_class_label = classifier(source_shared)

    		#source loss calculation
    		source_domain = loss_adv(source_domain_label, source_domainv_label)
    		loss += source_domain

    		source_class = loss_classification(source_class_label, source_classv_label)
    		loss += source_class

    		source_diff = loss_diff(source_private, source_shared)
    		loss += source_diff

    		source_mse = loss_recon1(source_rec, source_inputv_img)
    		loss += source_mse

    		source_simse = loss_recon2(source_rec, source_inputv_img)
    		loss += source_simse

    		loss.backward()
    		# if needed to update the lr, uncomment the following lines
    		# opt_model = exp_lr_scheduler(optimizer=opt_model, step=i)
    		# opt_discriminator = exp_lr_scheduler(optimizer=opt_discriminator, step=i)
    		# opt_classifier = exp_lr_scheduler(optimizer=opt_classifier, step=i)
    		opt_model.step()
        	opt_discriminator.step()
        	opt_classifier.step()

        	i += 1
            
        #print loss for each epoch

        #save the model per 50 epochs
        if epoch % 50 == 0:
        	torch.save(combined_model.state_dict(), param['output_dir'] + '/pda_epoch_' + str(epoch) + '.pth')
    

if __name__ == "__main__":
    # Read parameter values from the console
    parser = argparse.ArgumentParser(description = 'Domain Adaptation')
    parser.add_argument('--number_of_gpus', type = int, nargs = '?', default = '1', help = "Number of gpus to run")
    parser.add_argument('--network_name', type = str, default = 'ResNet50', help = "Name of the feature extractor network")
    parser.add_argument('--dataset_name', type = str, default = 'Office', help = "Name of the source dataset")
    parser.add_argument('--dropout_classifier', type = float, default = 0.25, help = "Dropout ratio for classifier")
    parser.add_argument('--dropout_discriminator', type = float, default = 0.25, help = "Dropout ratio for discriminator")    
    parser.add_argument('--source_path', type = str, default = 'amazon_31_list.txt', help = "Path to source dataset")
    parser.add_argument('--target_path', type = str, default = 'webcam_10_list.txt', help = "Path to target dataset")
    parser.add_argument('--lr_classifier', type = float, default = 0.0001, help = "Learning rate for classifier model")
    parser.add_argument('--b1_classifier', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for classifier model optimizer")
    parser.add_argument('--b2_classifier', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for classifier model optimizer")
    parser.add_argument('--lr_discriminator', type = float, default = 0.00001, help = "Learning rate for discriminator model")
    parser.add_argument('--b1_discriminator', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for discriminator model optimizer")
    parser.add_argument('--b2_discriminator', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for discriminator model optimizer")
    parser.add_argument('--lr_combined', type = float, default = 0.00001, help = "Learning rate for combined model")
    parser.add_argument('--b1_combined', type = float, default = 0.9, help = "Exponential decay rate of first moment \
                                                                                             for combined model optimizer")
    parser.add_argument('--b2_combined', type = float, default = 0.999, help = "Exponential decay rate of second moment \
                                                                                            for combined model optimizer")
    parser.add_argument('--classifier_loss_weight', type = float, default = 1, help = "Classifier loss weight")
    parser.add_argument('--discriminator_loss_weight', type = float, default = 4, help = "Discriminator loss weight")
    parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size for training")
    parser.add_argument('--test_interval', type = int, default = 3, help = "Gap between two successive test phases")
    parser.add_argument('--num_iterations', type = int, default = 12000, help = "Number of iterations")
    parser.add_argument('--snapshot_interval', type = int, default = 500, help = "Minimum gap between saving outputs")
    parser.add_argument('--output_dir', type = str, default = 'Models', help = "Directory for saving outputs")
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(list(np.arange(args.number_of_gpus))).strip('[]')

    # Initialize parameters
    param = {}
    param["number_of_gpus"] = args.number_of_gpus
    param["network_name"] = args.network_name
    param["inp_dims"] = [224, 224, 3]
    param["num_iterations"] = args.num_iterations
    param["lr_classifier"] = args.lr_classifier
    param["b1_classifier"] = args.b1_classifier
    param["b2_classifier"] = args.b2_classifier    
    param["lr_discriminator"] = args.lr_discriminator
    param["b1_discriminator"] = args.b1_discriminator
    param["b2_discriminator"] = args.b2_discriminator
    param["lr_combined"] = args.lr_combined
    param["b1_combined"] = args.b1_combined
    param["b2_combined"] = args.b2_combined        
    param["batch_size"] = int(args.batch_size/2)
    param["class_loss_weight"] = args.classifier_loss_weight
    param["dis_loss_weight"] = args.discriminator_loss_weight    
    param["drop_classifier"] = args.dropout_classifier
    param["drop_discriminator"] = args.dropout_discriminator
    param["test_interval"] = args.test_interval
    param["source_path"] = os.path.join("Data", args.dataset_name, args.source_path)
    param["target_path"] = os.path.join("Data", args.dataset_name, args.target_path)
    param["snapshot_interval"] = args.snapshot_interval
    param["output_path"] = os.path.join("Snapshot", args.output_dir)

    # Create directory for saving models and log files
    if not os.path.exists(param["output_path"]):
        os.mkdir(param["output_path"])
    
    # Load source and target data
    param["source_data"], param["source_label"] = data_loader(param["source_path"], param["inp_dims"])
    param["target_data"], param["target_label"] = data_loader(param["target_path"], param["inp_dims"])

    # Encode labels into one-hot format
    param["source_label"], param["target_label"] = one_hot_encoding(param)

    # Train data
    train(param)


