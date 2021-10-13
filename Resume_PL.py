# Progressive learning on quantization
from torch import nn, optim, cuda, device,no_grad
from utils_code import util_zhang as util_zhang, utils_PL, tensor_board_utils as tbutils
import numpy as np
from tqdm import tqdm
import os
from U_CapsNets.TUCaN_v1_2_PL import CapsNet_MR
from U_CapsNets.TUCaN_v1_2 import CapsNet_MR
from PIL import Image
from utils_code.utils import save_checkpoint, rfolder, isnan, reconstruction_loss, resume_model
import time
from apex import amp    #<==AMP
from tensorboardX import SummaryWriter
tb = False
model_path = 'Results_/RP_U_CapsNet_tiny_PL_Resume_imagenet/_100/BACKUP_model_log/checkpoint_RP_U_CapsNet_tiny_PL_Resume_65.pth.tar' #<-PL_IN
# model_path = 'Results_/RP_U_CapsNet_tiny_PL_TinyImagenet/_100_3/BACKUP_model_log/checkpoint_RP_U_CapsNet_tiny_PL_Resume_65.pth.tar'    #<-PL
# model_path = 'Results_/RP_parallel_U_CapsNet_tiny_TinyImagenet/_100/BACKUP_model_log/checkpoint_RP_parallel_U_CapsNet_tiny_90.pth.tar' #<- SSL
# model_path = 'Results_/FT_COCO_COCO_stuff/_100_2/BACKUP_model_log/checkpoint_FT_COCO_99.pth.tar' #<-PL_FT
# model_path = 'Results_/FT_COCO_COCO_stuff/_100/BACKUP_model_log/checkpoint_FT_COCO_99.pth.tar'    #<-PL_IN_FT
model_path = 'Results_/FT_COCO_COCO/_125/BACKUP_model_log/checkpoint_FT_COCO_115.pth.tar' #<- SSL_FT

file_model_name = "TUCaN_FT_COCO"
print(file_model_name)
db_used = 'COCO_stuff'
CUDA = 'cuda:0'
device = device(CUDA)
batch_size, n_epochs, epoch_val, lr_G = 32, 266, 5, 2e-4
if tb: writer = SummaryWriter(f'runs/{file_model_name}')


if db_used == 'Imagenet': dataloaders = util_zhang.load_Imagenet_dataset('/media/TBData3/Datasets/ImageNetOriginale', batch_size)
if db_used == 'TinyImagenet':  dataloaders = util_zhang.load_dataset('/media/TBData3/Datasets/tiny-imagenet-200/train',
                                                                     '/media/TBData3/Datasets/tiny-imagenet-200/val', batch_size)
if db_used == 'FT_COCO': dataloaders = util_zhang.load_dataset('/media/TBData3/Datasets/COCO_stuff/train', '/media/TBData3/Datasets/COCO_stuff/val', batch_size)

folder_results = rfolder("Results_/"+file_model_name+"_"+db_used, n_epochs)

generator = CapsNet_MR(128).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=lr_G)
# g_optimizer = optim.Adam([{'params': generator.conv_layer.parameters(), 'lr': lr_G},
#                 {'params': generator.primary_capsules.parameters(), 'lr': lr_G*10},
#                 {'params': generator.digit_capsules.parameters(), 'lr': lr_G*10},
#                 {'params': generator.reconstruction.parameters(), 'lr': lr_G},
#             ])

generator, _, epoch_start = resume_model(model_path, generator, g_optimizer, map_location=CUDA)
generator, g_optimizer = amp.initialize(generator, g_optimizer, opt_level="O1")

criterion = nn.CrossEntropyLoss().cuda()
criterionAB = nn.MSELoss()
encoder_q = utils_PL.Quantization_module()

(image_batch_val, target) = next(iter(dataloaders['val']))
img_l_rs_v = image_batch_val[1].to(device)

tic = time.time()
log_loss_G, mean_loss_GLQ, mean_loss_GLAB = [], [], []
with tqdm(total=n_epochs-epoch_start) as pbar:
    for epoch in range(epoch_start,n_epochs):
        log_loss, log_lossq, log_lossab = 0, 0, 0

        depth = encoder_q.depth_identifier(epoch)
        with cuda.amp.autocast(): #<==AMP
            for batch_id, (image_batch,target) in enumerate(dataloaders['train']):

                img_ab_rs = image_batch[0]
                img_l_rs = image_batch[1].to(device)
                img_rxs_ab = image_batch[2]

                g_optimizer.zero_grad()
                img_ab_pred,img_q = generator(img_l_rs ,depth=depth)

                targets, boost_nongray, img_ab_rs = encoder_q.forward(img_ab_rs,depth=depth)
                targets = targets.to(img_ab_pred.device)
                boost_nongray = boost_nongray.to(img_ab_pred.device)+ 1e-7

                img_ab_rs = img_ab_rs.to(img_ab_pred.device)
                lossAB = reconstruction_loss(img_ab_rs, img_ab_pred,criterionAB)
                lossQ = (criterion(img_q, targets) * boost_nongray.squeeze(1)).mean()
                loss = lossAB + lossQ
                with amp.scale_loss(loss, g_optimizer) as scaled_loss: #<==AMP
                    scaled_loss.backward() #<==AMP

                g_optimizer.step()
                log_loss += loss.item()
                log_lossq += lossQ.item()
                log_lossab += lossAB.item()
                del loss, img_ab_pred, img_q, targets, boost_nongray, \
                    image_batch, target, lossAB, lossQ, img_ab_rs, img_l_rs, img_rxs_ab
            log_loss_G.append(log_loss/(batch_id+1))
            mean_loss_GLQ.append(log_lossq/(batch_id+1))
            mean_loss_GLAB.append(log_lossab/(batch_id+1))
            #VALIDATION
            if epoch % epoch_val == 0:
                img_l_rs_v = image_batch_val[1].to(device)
                if tb:
                    writer.add_image('Colourisation',
                                     tbutils.plot_colourisation(generator, img_l_rs_v, batch_size, folder_results,
                                                                epoch),
                                     global_step=len(dataloaders['train']) * epoch)
                else:
                    img_ab_pred, _ = generator(img_l_rs_v)
                    for j in range(batch_size):
                        img_l_rs_v = img_l_rs_v.to(img_ab_pred.device)
                        img_rgb = util_zhang.postprocess_tens(img_l_rs_v, img_ab_pred, j, mode='bilinear')
                        im = Image.fromarray((img_rgb * 255).astype(np.uint8))
                        if not os.path.exists(os.path.join(folder_results, str(epoch))):
                            os.mkdir(os.path.join(folder_results, str(epoch)))
                        im.save(os.path.join(folder_results, str(epoch), "val_" + str(j) + ".jpeg"))
                    del img_ab_pred, img_rgb

                save_checkpoint({
                    'epoch': epoch + 1,
                    'depth': depth,
                    'loss_type': 'MSE',
                    'arch': file_model_name,
                    'state_dict': generator.state_dict(),
                    'optimizer': g_optimizer.state_dict(),
                }, os.path.join(folder_results,
                                "BACKUP_model_log/checkpoint_" + file_model_name + "_" + str(epoch) + ".pth.tar"))
                del img_l_rs_v
            toc = time.time()
            pbar.set_description(("{:.1f}s - loss: {:.3f} - Depth = {}".format((toc - tic), np.mean(log_loss_G), depth)))
            pbar.update(1)
            if tb:
                writer.add_scalars('Loss',
                                  {'G_loss': np.mean(log_loss_G), 'G_loss_LQ': np.mean(mean_loss_GLQ),
                                   'G_loss_LAB': np.mean(mean_loss_GLAB)},
                                  epoch * len(dataloaders['train']) + batch_id)

save_checkpoint({
        'epoch': epoch + 1,
        'depth': depth,
        'loss_type': 'MSE',
        'arch': file_model_name,
        'state_dict': generator.state_dict(),
        'optimizer': g_optimizer.state_dict(),
    }, os.path.join(folder_results,
                    "model_log/checkpoint_" + file_model_name + "_" + str(epoch) + ".pth.tar"))
