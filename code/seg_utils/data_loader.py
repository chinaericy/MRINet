import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
class Medical_Dataset(data.Dataset):
    def __init__(self, root, trainsize=512,mode='train',augmentation_prob=0.4):
        self.mode=mode
        self.trainsize = trainsize
        self.image_root = root+"image/"
        self.gt_root = root+"label1/"
        # print("gt_root",self.gt_root)
        # self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.png') ]
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        
        #if self.mode=="train" or self.mode=="val":
        #    self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png') ]        
        #    self.gts = sorted(self.gts)        
            #self.filter_files()
        self.size = len(self.images)
        
        self.img_transform = transforms.Compose([
            #transforms.Resize((self.trainsize,self.trainsize),Image.BILINEAR),
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        #self.gt_transform = transforms.Compose([
        #    #transforms.Resize((self.trainsize, self.trainsize), Image.NEAREST),
        #    transforms.ToTensor()
        #    ])

    def __getitem__(self, index):
        imageName=os.path.basename(self.images[index]).split(".")[0]
        gtFile=self.gt_root+imageName+".png"
        # gtFile = self.gt_root + imageName + ".jpg"

#原始的实现
        image = self.rgb_loader(self.images[index])
        image =  transforms.ToTensor()(image)*255

        if self.mode=="train" or self.mode=="val"or self.mode=="test":
            gt = self.binary_loader(gtFile)
            gt = transforms.ToTensor()(gt)*255
            #print(gt)

##自己的实现
#        image = self.rgb_loader(self.images[index])

        #print(index)
#        #image,gt = self.resize(image,gt)
#        #print("变换后的大小为",transforms.ToTensor()(gt).shape)

#        image = torch.tensor(image,dtype=torch.float64)
#        image=image.permute(2,0,1)
#        if self.mode=="train" or self.mode=="val":
#            gt = self.binary_loader(self.gts[index])
#            gt = torch.tensor(gt,dtype=torch.float64)
#            gt=gt.unsqueeze(0)
#        #print("image的大小为：：：：：：",gt.shape)
#        #gt=gt.permute(2,0,1)
############
        #print(torch.max(gt))

        #if self.mode=="train" or self.mode=="val":
            assert image.size()[-1] == self.trainsize #宽和高要一致
            assert image.size()[-2] ==self.trainsize
            assert gt.size()[-1] == self.trainsize #宽和高要一致
            assert gt.size()[-2] ==self.trainsize

            return image, gt
        elif self.mode=="predict":
            # print(self.images[index].split("/")[-1][:-len(".jpg")])
            return image, self.images[index].split("/")[-1][:-len(".png")]
            # return image, self.images[index].split("/")[-1][:-len(".jpg")]
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
           
            if img.size == gt.size:
            
                images.append(img_path)
                gts.append(gt_path)
            else:
                print(img_path,img.size,gt_path,gt.size)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img=img.convert('RGB')
            #img=np.array(img)#/255.0
            return img
        #data=np.fromfile(path,dtype=np.uint8)
        #img= cv2.imdecode(data,cv2.IMREAD_COLOR )
        #return img
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img=img.convert('L')
            # return img.convert('1')
            #img=np.array(img)
            return img
        
        #data=np.fromfile(path,dtype=np.uint8)
        #img= cv2.imdecode(data,0 )
        #return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(root, batchsize, trainsize, num_workers=4, mode='train',augmentation_prob=0.4,shuffle=True, pin_memory=True):

    dataset = Medical_Dataset(root= root, trainsize= trainsize,mode =mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


