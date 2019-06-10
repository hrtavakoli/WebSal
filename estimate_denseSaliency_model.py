'''
A sample script to show how to use the saliency model

@author: Hamed R. Tavakoli
'''
import re
import torch

from PIL import Image

from deepsalmodelRes import DeepSal

import torchvision.transforms as transforms

from utils import padded_resize, postprocess_predictions

import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.device(device)

height = 256
width = 320

out_height = 256
out_width =  320


out_height = 64
out_width =  80


def normalize_data(x):
    x = x.view(x.size(0), -1)
    x_max, idx = torch.max(x, dim=1, keepdim=True)
    x = x / (x_max.expand_as(x) + 1e-8)
    return x


class EstimateSaliency(object):

    def __init__(self, img_path, model_path):
        super(EstimateSaliency, self).__init__()
        self.impath = img_path
        self.model = DeepSal().to(device)

        self.model.eval()
        self.load_checkpoint(model_path)

    def load_checkpoint(self, model_path):
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)

            # '.'s are no longer allowed in module names, but pervious _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = checkpoint['state_dict']
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            self.model.load_state_dict(state_dict=state_dict, strict=True)
            print("=> loaded checkpoint '{}' )".format(model_path))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def estimate(self, savefolder):

        output_path = savefolder
        new_path = self.impath
        for file in os.listdir(new_path):

            if not file.endswith('.png'):
                continue

            imageName = file[:-4]
            imgO = Image.open(os.path.join(self.impath, file)).convert('RGB')
            orig_w = imgO.size[1]
            orig_h = imgO.size[0]

            imgO = padded_resize(imgO, height, width)

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

            img1 = transform(imgO)

            img1 = img1.to(device)

            img1 = torch.unsqueeze(img1, dim=0)

            saloutput = self.model(img1)
            saloutput = normalize_data(saloutput)
            saloutput = saloutput.view([1, out_height, out_width])
            saloutput = torch.squeeze(saloutput, 0)
            saloutput = saloutput.cpu().data.numpy()

            saloutput = postprocess_predictions(saloutput, orig_w, orig_h)
            a = 0.015*min(orig_w, orig_h)
            saloutput = (saloutput - np.min(saloutput)) / (np.max(saloutput) - np.min(saloutput))
            saloutput = gaussian_filter(saloutput, sigma=a)
            saloutput = (saloutput - np.min(saloutput)) / (np.max(saloutput) - np.min(saloutput))
            imgN = Image.fromarray((saloutput*255).astype(np.uint8))
            imgN.save('{}/{}.jpg'.format(output_path, imageName), 'JPEG')

if __name__ == "__main__":

    folder = '/ssd/rtavah1/web/test/'
    res_folder = './result/natmodel/'
    model_trainer = EstimateSaliency(img_path=folder,
                                     model_path='./web_model/model_best_256x320.pth.tar')
    model_trainer.estimate(savefolder=res_folder)
