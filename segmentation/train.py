import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder

import timm
from datasets.dataset import Smoke_datasets
from tensorboardX import SummaryWriter
from models.egeunet import EGEUNet

from engine import *
import os
import sys
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

def data_preproc(data_path, flag):
    dataset = Smoke_datasets(data_path, config, train=True) # actual smoke images
    if flag == 'org':
        subset_indices = np.random.choice(len(dataset), size=int(len(dataset)), replace=False)
    elif flag == 'smote':
        subset_indices = np.random.choice(len(dataset), size=int(len(dataset)), replace=False)
    subset = Subset(dataset, subset_indices)
    return subset


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    subset1 = data_preproc(config.data_path1, 'org')
    subset2 = data_preproc(config.data_path2, 'org')
    subset3 = data_preproc(config.data_path3, 'org')
    subset4 = data_preproc(config.data_path4, 'org')
    subset5 = data_preproc(config.data_path5, 'org')

    subset6 = data_preproc(config.data_path6, 'smote')
    subset7 = data_preproc(config.data_path7, 'smote')
    subset8 = data_preproc(config.data_path8, 'smote')
    subset9 = data_preproc(config.data_path9, 'smote')
    subset10 = data_preproc(config.data_path10, 'smote')

    combined_dataset = ConcatDataset([
        subset1, subset2, subset3, 
        subset4, subset5,
        subset6, subset7, subset8, subset9, subset10,
        ])
    
    # combined_dataset = ConcatDataset([subset3, subset4, subset5])
    # combined_dataset = Smoke_datasets(config.data_path1, config, train=True) 
    # Now we perform k-fold cross-validation on the combined_dataset

    k_folds = 5

    # Define the size of each fold
    fold_size = len(combined_dataset) // k_folds

    for fold in range(k_folds):

        resume_model = os.path.join(checkpoint_dir, "latest_k-fold1-500.pth")

        # Generate the indices for training and validation sets
        train_indices = list(range(0, fold*fold_size)) + list(range((fold+1)*fold_size, len(combined_dataset)))
        val_indices = list(range(fold*fold_size, (fold+1)*fold_size))

        # train_indices = list(range(0, int(0.7*fold_size)))
        # val_indices = list(range(int(0.7*fold_size), fold_size))

        # Generate the samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(combined_dataset, 
                                    sampler=train_sampler,
                                    batch_size=config.batch_size, 
                                    pin_memory=True,
                                    num_workers=config.num_workers)
        val_loader = DataLoader(combined_dataset,
                                    sampler=val_sampler,
                                    batch_size=1,
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

        print('#----------Prepareing loss, opt, sch and amp----------#')
        criterion = config.criterion
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)

        print('#----------Set other params----------#')
        min_loss = 999
        start_epoch = 1
        min_epoch = 1

        if os.path.exists(resume_model):
            print('#----------Resume Model and Other params----------#')
            checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            saved_epoch = checkpoint['epoch']
            start_epoch += saved_epoch
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
            logger.info(log_info)

        step = 0
        print('#----------Training----------#')
        for epoch in range(start_epoch, config.epochs + 1):

            torch.cuda.empty_cache()

            step = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                step,
                logger,
                config,
                writer
            )

            loss = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config
                )

            if loss < min_loss:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, '{}_best_k-fold{}.pth'.format(config.ckpt_name,str(fold+1))))
                min_loss = loss
                min_epoch = epoch

            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, '{}_latest_k-fold{}.pth'.format(config.ckpt_name, str(fold+1)))) 

        if os.path.exists(os.path.join(checkpoint_dir, '{}_best_k-fold{}.pth'.format(config.ckpt_name, str(fold+1)))):
            print('#----------Testing----------#')
            best_weight = torch.load(config.work_dir + 'checkpoints/{}_best_k-fold{}.pth'.format(config.ckpt_name, str(fold+1)), map_location=torch.device('cpu'))
            model.load_state_dict(best_weight)
            loss = test_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    logger,
                    config,
                )
            os.rename(
                os.path.join(checkpoint_dir, '{}_best_k-fold{}.pth'.format(config.ckpt_name, str(fold+1))),
                os.path.join(checkpoint_dir, 'best-epoch{}-loss{:4f}-k-fold{}.pth'.format(min_epoch, min_loss, str(fold+1)))
            )      


if __name__ == '__main__':
    config = setting_config
    main(config)