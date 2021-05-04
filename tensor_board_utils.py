import numpy as np
import torch
import matplotlib.pyplot as plt
from utils_code import util_zhang

# constant for classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output,mask = net(images)
    # convert output probabilities to predicted class
    # preds_tensor = mask #torch.max(mask, 1)
    # preds = np.squeeze(preds_tensor.cpu().detach().numpy())
    # return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    return output, mask


def plot_classes_preds(net, images, labels, batch_size):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    max_idx = np.argmax(probs.cpu().numpy(), 1)
    fig = plt.figure(figsize=(batch_size/4, 4))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(int(batch_size/4),4,idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx].cpu(), one_channel=True)
        ax.set_title("{0}\n(label:{1})".format(
            classes[max_idx[idx]],
            classes[labels[idx]]),
                    color=("green" if max_idx[idx]==labels[idx].item() else "red"))
    return fig

from PIL import Image

def get_concat_h(imlist):
    width = 0
    for im in imlist: width += im.width
    dst = Image.new('RGB', (width, im.height))
    for i,im in enumerate(imlist):
        dst.paste(im, (i*im.width, 0))
    return dst

def get_concat_v(imlist):
    height = 0
    for im in imlist: height += im.height
    dst = Image.new('RGB', (im.width, height))
    for i,im in enumerate(imlist):
        dst.paste(im, (0, i*im.height))

    return dst

import os
def plot_colourisation(net, img_l_rs_v, batch_size, fr, epoch,  depth = None):
    if depth==None: img_ab_pred, _ = net(img_l_rs_v)
    else: img_ab_pred, _ = net(img_l_rs_v, depth=depth)

    images_list = []
    for j in range(batch_size):
        img_rgb = util_zhang.postprocess_tens(img_l_rs_v, img_ab_pred, j, mode='bilinear')
        im = Image.fromarray((img_rgb * 255).astype(np.uint8))
        if not os.path.exists(os.path.join(fr, str(epoch))):
            os.mkdir(os.path.join(fr, str(epoch)))
        im.save(os.path.join(fr, str(epoch), "val_" + str(j) + ".jpeg"))
        images_list.append(im)

    row = int(batch_size/5)
    row_im = []
    for i in range(row):
        j = (i+1)*5
        k = i*5
        row_im.append(get_concat_h(images_list[k:j]))
    fig = get_concat_v(row_im)
    return torch.from_numpy(np.array(fig, dtype=np.float).transpose(2, 0, 1) / 255)



