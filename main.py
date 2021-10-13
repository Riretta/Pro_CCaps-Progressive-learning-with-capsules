import utils_code.util_zhang as util_zhang
from utils_code.utils_PL import *
from utils_code.utils import *
import time
from tqdm import tqdm
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from torch import device, cuda, optim
""" SETTING PARAMETER """
additional_appendix = "_ProCCaps_"
routes = (32,9,9)
size = 224
save_inFolder = True                                                                                            #savelog
r = False                                                                                                        #resume

n_cuda = 0                                                                                               #cuda_id
batch_size, n_epochs, epoch_start, epoch_val = 64, 65, 0, 5
db_used = 'Imagenet'
CUDA = 'cuda:'+str(n_cuda)
path_resume_model = 'Results_/**/checkpoint_ProCCaps_100.pth.tar'
"""------------------------"""

from U_CapsNets.Pro_CCaps import CapsNet_MR
file_model_name = "Model_" +additional_appendix

if additional_appendix == '_UNET_':
    epoch_start = 16
    from U_CapsNets.U_Net_PL import CapsNet_MR
    file_model_name = "U_NET_PL"

print(f"model - {file_model_name} - dataset {db_used}")

## SET device
device = device(CUDA)
cuda.set_device(device)

## load dataset
if db_used == 'Imagenet': dataloaders = util_zhang.load_Imagenet_dataset('/media/**/Datasets/ImageNetOriginale', batch_size, size=size)
if db_used == 'TinyImagenet':  dataloaders = util_zhang.load_dataset('/media/**/Datasets/tiny-imagenet-200/train',
                                                                     '/media/**/Datasets/tiny-imagenet-200/val', batch_size, size=size)
## create data log folder
if save_inFolder: folder_results = rfolder("Results_/"+file_model_name+"_"+db_used, n_epochs)
#------------------------------------------------------------------------------------------------------------#
## model optimizer
if not additional_appendix=="_UNET_": generator = CapsNet_MR(128,num_routes=routes)
else: generator = CapsNet_MR()

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
if r:
    generator, g_optimizer, epoch_start = resume_model(path_resume_model,generator,g_optimizer,map_location='cuda')
    for state in g_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, Tensor):
                state[k] = v.to(device)

generator = generator.to(device)

criterion = nn.CrossEntropyLoss().cuda()
criterionAB = nn.MSELoss()
encoder_q = Quantization_module()

## test batch
(image_batch_val, target) = next(iter(dataloaders['val']))
img_l_rs_v = image_batch_val[1].to(device)
print(f'val image size {img_l_rs_v.size()}')
## TRAINING
tic = time.time()
log_loss_G = []
with tqdm(total=n_epochs-epoch_start) as pbar:
    for epoch in range(epoch_start, n_epochs):
        log_loss = 0
        depth = encoder_q.depth_identifier(epoch)
        for batch_id, (image_batch,target) in enumerate(dataloaders['train']):

            img_ab_rs = image_batch[0]
            img_l_rs = image_batch[1].to(device)
            img_rxs_ab = image_batch[2]

            g_optimizer.zero_grad()
            img_ab_pred, img_q = generator(img_l_rs,depth)
            targets, boost_nongray, img_ab_rs = encoder_q.forward(img_ab_rs, depth=depth,mode=size,UNET_mode=True)
            targets = targets.to(img_ab_pred.device)
            boost_nongray = boost_nongray.to(img_ab_pred.device)

            img_ab_rs = img_ab_rs.to(img_ab_pred.device)

            if additional_appendix == "_onlyAB_":
                lossAB = reconstruction_loss(img_ab_rs, img_ab_pred, criterionAB)
                loss = lossAB
            elif additional_appendix == "_onlyQ_":
                lossQ = (criterion(img_q, targets) * boost_nongray.squeeze(1)).mean()
                loss = lossQ
            else:
                lossAB = reconstruction_loss(img_ab_rs, img_ab_pred, criterionAB)
                lossQ = (criterion(img_q, targets) * boost_nongray.squeeze(1)).mean()
                loss = lossAB + lossQ
            loss.backward()
            g_optimizer.step()
            log_loss += loss.item()

        log_loss_G.append(log_loss/batch_id)
            # VALIDATION
        if epoch % epoch_val == 0 and save_inFolder:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': file_model_name,
                'state_dict': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
            }, os.path.join(folder_results,
                            "BACKUP_model_log/checkpoint_" + file_model_name + "_" + str(epoch) + ".pth.tar"))
            img_l_rs_v = image_batch_val[1].to(device)
            img_ab_pred, _ = generator(img_l_rs_v,depth)
            for j in range(batch_size):
                img_l_rs_v = img_l_rs_v.to(img_ab_pred.device)
                img_rgb = util_zhang.postprocess_tens(img_l_rs_v, img_ab_pred, j, mode='bilinear')
                im = Image.fromarray((img_rgb * 255).astype(np.uint8))
                if not os.path.exists(os.path.join(folder_results, str(epoch))):
                    os.mkdir(os.path.join(folder_results, str(epoch)))
                im.save(os.path.join(folder_results, str(epoch), "val_" + str(j) + ".jpeg"))

            del img_ab_pred, img_rgb, img_l_rs_v
        toc = time.time()
        pbar.set_description(
            ("{:.1f}s - loss: {:.3f} - Depth = {}".format((toc - tic), log_loss_G[-1], depth)))
        pbar.update(1)

save_checkpoint({
    'epoch': epoch + 1,
    'depth':depth,
    'loss_type': 'MSE',
    'arch': file_model_name,
    'state_dict': generator.state_dict(),
    'optimizer': g_optimizer.state_dict(),
},os.path.join(folder_results,"model_log/checkpoint_"+file_model_name+"_"+str(epoch)+".pth.tar"))


import matplotlib.pyplot as plt
epochs_G = np.arange(1, len(log_loss_G)+1)
plt.plot(epochs_G, log_loss_G, color='g', label='loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training phase')
plt.savefig(os.path.join(folder_results,"log_loss_G.png"))
plt.clf()
