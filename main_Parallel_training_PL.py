# Progressive learning on quantization
from torch import nn, optim, cuda, device
from utils_code import util_zhang as util_zhang, utils_PL, tensor_board_utils as tbutils
import numpy as np
from tqdm import tqdm
import os
from U_CapsNets.TUCaN_v1_PL import CapsNet_MR
from PIL import Image
from utils_code.utils import save_checkpoint, rfolder, reconstruction_loss
import matplotlib.pyplot as plt
import time
# from apex import amp    #<==AMP
from tensorboardX import SummaryWriter

#Parallelized training of Colorization U_CapsNet_Niki (forward training in self-supervision)
tb = False
# default `log_dir` is "runs" - we'll be more specific here

file_model_name = "TUCaN_2_PL"
print(file_model_name)
db_used = 'TinyImagenet'
n_cuda = 0
CUDA = 'cuda:'+str(n_cuda)
device = device( CUDA)
batch_size, n_epochs,epoch_val, lr_G = 32, 60, 5, 2e-4
if tb:
    writer = SummaryWriter(f'runs/{file_model_name}')

if db_used == 'Imagenet': dataloaders = util_zhang.load_Imagenet_dataset('/media/TBData3/Datasets/ImageNetOriginale', batch_size)
if db_used == 'TinyImagenet':  dataloaders = util_zhang.load_dataset('/media/TBData3/Datasets/tiny-imagenet-200/train',
                                                                     '/media/TBData3/Datasets/tiny-imagenet-200/val', batch_size)

folder_results = rfolder("Results_/"+file_model_name+"_"+db_used, n_epochs)
cuda.set_device(device)
generator = nn.DataParallel(CapsNet_MR(128), device_ids=[n_cuda])
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
# g_optimizer = optim.Adam([{'params': generator.conv_layer.parameters(), 'lr': lr_G},
#                 {'params': generator.primary_capsules.parameters(), 'lr': lr_G*10},
#                 {'params': generator.digit_capsules.parameters(), 'lr': lr_G*10},
#                 {'params': generator.reconstruction.parameters(), 'lr': lr_G},
#             ])
# generator, g_optimizer = amp.initialize(generator, g_optimizer, opt_level="O1")

criterion = nn.CrossEntropyLoss().cuda()
criterionAB = nn.MSELoss()
encoder_q = utils_PL.Quantization_module()

(image_batch_val, target) = next(iter(dataloaders['val']))
img_l_rs_v = image_batch_val[1].to(device)
tot_batch = len(dataloaders['train'])
tic = time.time()
log_loss_G, mean_loss_GLQ, mean_loss_GLAB = [], [], []
with tqdm(total=n_epochs) as pbar:
    generator.train()
    for epoch in range(n_epochs):
        log_loss, log_lossq, log_lossab = 0, 0, 0
        depth = encoder_q.depth_identifier(epoch) #<--identify the depth in progressive learning
        # with cuda.amp.autocast(): #<==AMP
        for batch_id, (image_batch,target) in enumerate(dataloaders['train']):
            if batch_id<10:
                batch_id = 0
                img_ab_rs = image_batch[0]
                img_l_rs = image_batch[1].to(device)
                img_rxs_ab = image_batch[2]

                g_optimizer.zero_grad()
                img_ab_pred,img_q = generator(img_l_rs,depth=depth)
                print(img_ab_pred.size())
                targets, boost_nongray, img_ab_rs = encoder_q.forward(img_ab_rs,depth=depth)
                targets = targets.to(img_ab_pred.device)
                boost_nongray = boost_nongray.to(img_ab_pred.device)

                img_ab_rs = img_ab_rs.to(img_ab_pred.device)
                lossAB = reconstruction_loss(img_ab_rs, img_ab_pred,criterionAB)
                print(img_ab_rs.size())
                lossQ = (criterion(img_q, targets) * boost_nongray.squeeze(1)).mean()
                loss = lossAB + lossQ
                print(loss)

                loss.backward()
                g_optimizer.step()
                log_loss += loss.item()
                log_lossq += lossQ.item()
                log_lossab += lossAB.item()
                if batch_id % 20000 == 0:
                    toc = time.time()
                    print(f'batch: {batch_id} of {tot_batch}: time  {int((toc-tic)/60)}m,{int((toc-tic)%60)}s')
                del loss, img_ab_pred, img_q, targets, boost_nongray,\
                    image_batch, target, lossAB, lossQ, img_ab_rs, img_l_rs, img_rxs_ab
            else:
                break
        log_loss_G.append(log_loss/(batch_id+1))
        mean_loss_GLQ.append(log_lossq/(batch_id+1))
        mean_loss_GLAB.append(log_lossab/(batch_id+1))
        #VALIDATION
        if epoch%epoch_val == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'depth': depth,
                'loss_type': 'MSE',
                'arch': file_model_name,
                'state_dict': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
            }, os.path.join(folder_results,
                            "BACKUP_model_log/checkpoint_" + file_model_name + "_" + str(epoch) + ".pth.tar"))
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
    'depth':depth,
    'loss_type': 'MSE',
    'arch': file_model_name,
    'state_dict': generator.state_dict(),
    'optimizer': g_optimizer.state_dict(),
},os.path.join(folder_results,"model_log/checkpoint_"+file_model_name+"_"+str(epoch)+".pth.tar"))

epochs_G = np.arange(1, len(log_loss_G)+1)
plt.plot(epochs_G, log_loss_G, color='g', label='loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training phase')
plt.savefig(os.path.join(folder_results,"log_loss_G.png"))
plt.clf()




