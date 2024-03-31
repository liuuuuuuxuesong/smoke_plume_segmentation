import os
import sys
import numpy as np
from tqdm import tqdm
import timm
from glob import glob
from PIL import Image


import torch
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter


from sklearn.metrics import confusion_matrix

from datasets.dataset import Smoke_datasets, Smoke_test_datasets
from models.egeunet import EGEUNet

from engine import *
from utils import *

from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def save_masks(model_name, config, mask, output_path, image_file, palette=None):
	# Saves the image, the model output and the results after the post processing
    image_file = os.path.basename(image_file).split('.')[0]
    # print("mask shape", mask.shape)
    mask = np.where(np.squeeze(mask, axis=0) > 0.5, 1, 0)
    mask = mask.astype(np.uint8) * 255
    output_dir = os.path.join(output_path, config.infer_data_path.split('/')[-2]) # [-1] empty('')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mask = Image.fromarray(mask)
    save_path = os.path.join(output_dir, image_file+'.png')
    mask.save(save_path)

def infer_mask(test_loader,
                    model,
                    criterion,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    print("infer_mask function.......")

    model_name = "EGE-UNet"

    with torch.no_grad():
        for _, (img, msk, img_path) in enumerate(test_loader):
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            _, out = model(img)

            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            if not os.path.exists(config.pred_save_dir): os.makedirs(config.pred_save_dir)
            save_masks(model_name, config, out, config.pred_save_dir, img_path[0])

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    checkpoint_dir = os.path.join("ckpts", config.ckpt_name, 'checkpoints')

    outputs = os.path.join("infer_results", config.work_dir, 'outputs')
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = Smoke_test_datasets(config.infer_data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'], 
                        input_channels=model_cfg['input_channels'], 
                        c_list=model_cfg['c_list'], 
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        )
    else: raise Exception('network in not right!')
    model = model.cuda()

    criterion = GT_BceDiceLoss(wb=1, wd=1)

    print("checkpoint_dir", checkpoint_dir)

    ckpt_name = "sizes_all_best_k-fold1.pth"

    if os.path.exists(os.path.join(checkpoint_dir, ckpt_name)):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, ckpt_name), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        print("infer_mask starting.............")
        infer_mask(val_loader, model, criterion, config,)

if __name__ == '__main__':
    config = setting_config
    main(config)