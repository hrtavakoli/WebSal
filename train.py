'''
Script to train the dense saliency model

@author: Hamed R. Tavakoli
'''
import re
import os
import sys
import shutil


import torch
import torch.optim as optim
import torchvision.transforms as transforms
from database import SalDB
from deepsalmodelRes import DeepSal


from lossfunctions import KLLoss, NEGNSSLoss, ACCLoss

device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
torch.device(device)


learning_rate = 1e-8

height_dim = 256
width_dim = 320
ts = (64, 80)


class TrainSal(object):

    def __init__(self, batch_size, num_workers, root_folder):
        super(TrainSal, self).__init__()

        self.model = DeepSal().to(device)
        self.model.train()

        self.val_loss = 0.0

        self.batch_size = batch_size
        self.num_workers = num_workers
        transform_1 = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

        transform_target = transforms.Compose([transforms.Grayscale(1),
                                               transforms.ToTensor()])

        self.train_db = SalDB(root_folder=root_folder, input_size=(height_dim, width_dim),
                              output_size=ts, fold='train',
                              input_transform=transform_1,
                              target_transform=transform_target)
        self.valid_db = SalDB(root_folder=root_folder, input_size=(height_dim, width_dim),
                              output_size=ts, fold='val',
                              input_transform=transform_1,
                              target_transform=transform_target)

        parameters = self.model.parameters()
        self.optimizer = optim.Adam(parameters, lr=learning_rate,
                                    weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=1,
                                                              verbose=True)

        self.criterion_nss = NEGNSSLoss().to(device)
        self.criterion_kld = KLLoss().to(device)
        self.criterion_acc = ACCLoss().to(device)

    def _disable_main_trunk_params(self):
        for param in self.model.encode_image.parameters():
            param.requires_grad = False

    def load_checkpoint(self, model_path):
        # support densenet and pytorch 0.4 added
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

            self.model.load_state_dict(state_dict=state_dict, strict=False)
            print("=> loaded checkpoint '{}' )".format(model_path))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def save_checkpoint(self, is_best, filename='checkpoint_{}x{}.pth.tar'.format(height_dim, width_dim), prefix=''):

        state = {'state_dict': self.model.state_dict(),
                 'optimizero': self.optimizer}
        torch.save(state, prefix + filename)
        if is_best:
            shutil.copyfile(prefix + filename, prefix + 'model_best_{}x{}.pth.tar'.format(height_dim, width_dim))

    def train_val_loop(self, epoch, fold):

        if fold == 'val':
            dbl = torch.utils.data.DataLoader(self.valid_db, batch_size=self.batch_size,
                                              shuffle=False, num_workers=self.num_workers)
            self.model.eval()
            torch.set_grad_enabled(False)
        if fold == 'train':
            dbl = torch.utils.data.DataLoader(self.train_db, batch_size=self.batch_size,
                                              shuffle=True, num_workers=self.num_workers)
            self.model.train()
            torch.set_grad_enabled(True)

        running_loss = 0.0

        data_iterator = iter(dbl)

        for it in range(len(dbl)):

            img_id, img, map, fix = data_iterator.next()
            img = img.to(device)
            map = map.to(device)
            fix = fix.to(device)

            saloutput = self.model(img)
            loss1 = self.criterion_nss(saloutput, fix)
            loss2 = self.criterion_kld(saloutput, map)
            loss3 = self.criterion_acc(saloutput, map)
            loss = 7*loss1 + loss2 + 2*loss3

            if torch.isnan(loss):
                print('\nerror\n')
            if fold == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

            sys.stdout.write("\rEpoch %d -- %s %.01f%% -- Loss: %.03f" %
                            (epoch+1, fold, (it + 1) / len(dbl) * 100, running_loss / ((it + 1)*self.batch_size)))
            sys.stdout.flush()

        sys.stdout.write(" \n ")
        sys.stdout.flush()
        if fold == 'val':
            self.val_loss = running_loss / (it + 1)

    def train_val_model(self, num_epochs, log_dir, model_path=None):

        if model_path is not None:
            self.load_checkpoint(model_path)

        loss_value = 0
        for epoch in range(num_epochs):
            self.train_val_loop(epoch, 'train')
            self.train_val_loop(epoch, 'val')
            self.scheduler.step(self.val_loss)
            is_best = False
            if epoch > 0:
                if self.val_loss <= loss_value:
                    loss_value = self.val_loss
                    is_best = True
            else:
                loss_value = self.val_loss
                is_best = True
            self.save_checkpoint(is_best, prefix=log_dir)


if __name__ == "__main__":

    folder = '/mnt/Databases/websal/data/'
    model_trainer = TrainSal(batch_size=4, num_workers=2, root_folder=folder)
    model_trainer.train_val_model(20, './web_model/', './log_res50/model_best_256x320.pth.tar')