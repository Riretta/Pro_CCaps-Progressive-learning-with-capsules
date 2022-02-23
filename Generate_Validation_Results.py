"""model definition"""
"""trained not GAN"""

from U_CapsNets.TUCaN_v1_PL import CapsNet_MR
# ------------
from torch import Tensor, device
from torch.utils.data import DataLoader
from torchvision import transforms
from Zhang_github.data_imagenet import  ValImageFolder
import utils_code.util_zhang as util_zhang
from utils_code.utils import *
import os
from PIL import Image, ImageFile
import numpy as np
import shutil
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

additional_appendix = "ProCCaps"
routes = (32, 9, 9)
size = 224
root = 'test_dataset'
if os.path.exists(root):
  os.mkdir(root)
class RGB2LAB(object):
    def __init__(self):
        super(RGB2LAB,self).__init__()
    def __call__(self,img):
        (tens_rs_ab, tens_rs_l, tens_rxs_ab)= util_zhang.preprocess_img(img, HW=(size,size), resample=3)
        return tens_rs_ab, tens_rs_l, tens_rxs_ab
original_transform = transforms.Compose([RGB2LAB()])

"""Dataset to use"""
# val_dir_name, results_folder = '/media/TBData2/Datasets/COCO_stuff/val', "COCO_stuff"
val_dir_name, results_folder = '/media/TBData2/Datasets/ImageNet10k/Test', "Image10k_tiny"
# val_dir_name, results_folder = '/media/TBData2/Datasets/Places205/Test', "Places205"

"""batch_size"""
batch_size = 32
"""Init and load dataset"""
ImageDataset = ValImageFolder(val_dir_name, transform=original_transform)
dataloaders = DataLoader(ImageDataset, batch_size=batch_size, shuffle=False)

"""if you want to keep original dimension (enlarge the output of the model), you need orig_dim = True"""
orig_dim = False
extention, model_folder = 'png', "PL"
n_cuda, epoch = 0, str(65)
CUDA = "cuda:"+str(n_cuda)
device = device(CUDA)

"""Init model: the SSL models are in dataParallel box, the PL models are without dataParallel"""
model = CapsNet_MR(128, num_routes=routes)
depth = 'Q'
print(f"*The depth is {depth} ! Be aware of that*")

"""Checkpoint locations"""
model_path, appendix = os.path.join('models/WACV_model/checkpoint_ProCCaps_ImageNet_65.pth.tar'), "_wacv"  #<-PL_IN*

"""Resume pretrained model"""
model, epoch_start = resume_model(model_path, model, map_location=CUDA)
model = model.to(device)

"""Create folder for the coloured images"""
if not os.path.exists(os.path.join(root,results_folder)):
    os.mkdir(os.path.join(root,results_folder))
mother_folder = os.path.join(root,results_folder,epoch+"_"+model_folder+"_"+appendix)
mother_folder_orig_clone = mother_folder + '_orig_clone'
if os.path.exists(mother_folder):
    shutil.rmtree(mother_folder, ignore_errors=True)
if os.path.exists(mother_folder_orig_clone):
    shutil.rmtree(mother_folder_orig_clone, ignore_errors=True)
os.mkdir(mother_folder)
os.mkdir(mother_folder_orig_clone)



psnr_sum, psnr_sum_self = [], []
model.train()
for batch_id, (image_batch,index, target, name_path) in enumerate(tqdm(dataloaders)):
    ##extract the greyscale image
        img_ab_rs = image_batch[0].to(device)
        img_l_rs = image_batch[1].to(device)
        img_l_rs_v = img_l_rs.to(device)
        ##infer the colourisation channels
        if model_folder == 'SSL': img_ab_pred,_ = model(img_l_rs_v)
        elif model_folder == 'PL': img_ab_pred = model(img_l_rs_v, depth=depth)
        ##print check
        if batch_id == 0: print(img_ab_pred[0,:,:,:])
        ##for loop for image recomposition
        for j in range(len(img_l_rs_v)):
            if orig_dim:
                tens_orig = util_zhang.original_l(ImageDataset, index[j])
                L = tens_orig.unsqueeze(0).to(device)
                HW = tens_orig.shape[2:]
            else:
                L =  img_l_rs_v
                HW = img_ab_pred.shape[2:]

            img_rgb = util_zhang.postprocess_tens(L, img_ab_pred, j, mode='bilinear',HW_request=HW) #HW_request=img_ab_pred.shape[2:])#
            img_rgb = (img_rgb*255).astype(np.uint8)
            img_rgb_orig = util_zhang.postprocess_tens(L, img_ab_rs, j, mode='bilinear',HW_request=HW) #HW_request=img_ab_pred.shape[2:])#HW)
            img_rgb_orig = (img_rgb_orig * 255).astype(np.uint8)
            ##PSNR
            psnr_sum.append(psnr(img_rgb_orig, img_rgb))
            #Save output
            pil_image = Image.fromarray(img_rgb, 'RGB')
            if isinstance(target[j],Tensor):
                target_j = str(target[j].item())
            else:
                target_j = target[j]

            dstfolder = os.path.join(mother_folder, target_j)
            if not os.path.exists(dstfolder):
                os.mkdir(dstfolder)
                os.mkdir(os.path.join(mother_folder_orig_clone, target_j))
            pil_image.save(os.path.join(dstfolder, name_path[j].replace(extention,'png')))
            Image.fromarray(img_rgb_orig, 'RGB').save(os.path.join(mother_folder_orig_clone,
                                                                   target_j,
                                                                  name_path[j].replace(extention,'png')))
            del pil_image,img_rgb_orig,img_rgb,dstfolder#, tens_orig
        del image_batch,target,name_path
print(np.mean(psnr_sum))
