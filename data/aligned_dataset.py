import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
    
#####################################################################################################
class AlignedDataset(BaseDataset):
    def __init__(self,opt, 
                 transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()]),
                 mode='train'):
        self.opt = opt
        self.base_path = opt.dataroot
        self.mode = mode
        self.rgb_imgNames = [i for i in sorted(os.listdir(os.path.join(self.base_path, "RGB",self.mode))) if  i.endswith('.jpg')]
        self.thermal_imgNames = [i for i in sorted(os.listdir(os.path.join(self.base_path, "thermal",self.mode))) if i.endswith('.jpg') ] 
        
        self.mode = mode
        self.transform = transform
        
    
    def __getitem__(self, idx):

        # print("len of rgb and thermal"/,len(self.rgb_imgNames),len(self.thermal_imgNames))
        # if self.rgb_imgNames[idx].endswith('.jpg'):
            
        rgb_imName = self.rgb_imgNames[idx]
        # if self.rgb_imgNames[idx].endswith('.jpg'):
            
        thermal_imName = self.thermal_imgNames[idx]
        a_path = self.base_path, "RGB",self.mode,rgb_imName
        b_path = self.base_path, "thermal",self.mode, thermal_imName
        
        rgb = Image.open(os.path.join(self.base_path, "RGB",self.mode,rgb_imName))
        thermal = Image.open(os.path.join(self.base_path, "thermal",self.mode, thermal_imName))
        
        rgb_tf = self.transform(rgb)
        thermal_tf = self.transform(thermal)
        
        return {"A": rgb_tf, "B": thermal_tf, "A_paths": rgb_imName,"B_paths":thermal_imName}
        
    
    
    def __len__(self):
        return len(self.rgb_imgNames)

