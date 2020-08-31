import torch
import torch.nn as nn
from functions import ReverseLayerF
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )



class DSN(nn.Module):
    def __init__(self, code_size=100, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        self.base_model = models.resnet50(pretrained=True)
        self.base_layers=list(self.base_model.children())[:-1]
        self.source_encoder_conv = nn.Sequential(*self.base_layers)

        self.source_encoder_fc = nn.Sequential()
        self.source_encoder_fc.add_module('fc_pse', nn.Linear(in_features=512, out_features=code_size))
        self.source_encoder_fc.add_module('ac_pse', nn.ReLU(True))

        #########################################
        # private target encoder
        #########################################

        self.base_model2 = models.resnet50(pretrained=True)
        self.base_layers2=list(self.base_model2.children())[:-1]
        self.target_encoder_conv = nn.Sequential(*self.base_layers2)
        

        self.target_encoder_fc = nn.Sequential()
        self.target_encoder_fc.add_module('fc_pte', nn.Linear(in_features=512, out_features=code_size))
        self.target_encoder_fc.add_module('ac_pte', nn.ReLU(True))

        ################################
        # shared encoder (dann_mnist)
        ################################


        self.base_model3 = models.resnet50(pretrained=True)
        self.base_layers3=list(self.base_model3.children())[:-1]
        self.shared_encoder_conv = nn.Sequential(*self.base_layers3)
        
        self.shared_encoder_fc = nn.Sequential()
        self.shared_encoder_fc.add_module('fc_pshe', nn.Linear(in_features=512, out_features=code_size))
        self.shared_encoder_fc.add_module('ac_pshe', nn.ReLU(True))

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_she4', nn.Linear(in_features=code_size, out_features=100))
        self.shared_encoder_pred_class.add_module('relu_she4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_she5', nn.Linear(in_features=100, out_features=n_class))

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=100, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=100, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=code_size, out_features=588))
        self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))

        self.shared_decoder_conv = nn.Sequential()
        self.shared_decoder_conv.add_module('conv_sd2', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd2', nn.ReLU())

        self.shared_decoder_conv.add_module('us_sd2_u', nn.Upsample(scale_factor=2))

        self.shared_decoder_conv.add_module('conv_sd3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd3', nn.ReLU())

        self.shared_decoder_conv.add_module('us_sd4', nn.Upsample(scale_factor=2))

        self.shared_decoder_conv.add_module('conv_sd5', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd5', nn.ReLU(True))

        self.shared_decoder_conv.add_module('relu_sd5_u', nn.Upsample(scale_factor=2))

        self.shared_decoder_conv.add_module('conv_sd6', nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3,
                                                                  padding=1))


    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 512)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':

            # target private encoder
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 512)
            private_code = self.target_encoder_fc(private_feat)

        result.append(private_code)

        # shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 512)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 3, 16, 16)
        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result





