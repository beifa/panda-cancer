import os
import torch
import time
import random
import albumentations as A
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import torch.nn as nn
from hub_models import HUB_MODELS
from torch.optim import Adam, SGD
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler 
from dataset import trainDataset_insta, trainDataset_pkl, trainDataset_npy

#nightly
from torch.cuda import amp

DEVICE = torch.device('cuda')

path_log = 'C:\\Users\\pka\\kaggle\\panda\\log'
path_model = 'C:\\Users\\pka\\kaggle\\panda\\model' 
path_data = 'C:\\Users\\pka\\kaggle\\panda\\input'
path_checkpoint = 'C:\\Users\\pka\\kaggle\\panda\\checkpoint'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def loss_func(target, pred, func, scaler=None, opt= None):
    loss = func(pred, target)
    if opt is not None:
        # with apex.amp.scale_loss(loss, opt) as scaled_loss: 
        #     scaled_loss.backward()
        # loss.backward()
        # opt.step()
        # opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()  
    #loss return tensor
    return loss #loss.item()

def train_epoch(model, dataloader, func, scaler, opt):
    losses =[]
    preds = []
    labels = []
    for img, label in tqdm(dataloader):        
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        if opt is not None:
            opt.zero_grad()
            with amp.autocast():
                y_ = model(img)
                loss = loss_func(label, y_, func, scaler, opt)
        else:
            y_ = model(img)
            loss = loss_func(label, y_, func)

        pred = y_.sigmoid().sum(1).detach().round()
        preds.append(pred)
        losses.append(loss.detach().cpu().numpy())
        labels.append(label.sum(1))    
    if opt is None:
        #losses = np.mean(losses)
        return losses, preds, labels
    else:
        return losses


def main():
    fold = 0
    epoch = 3   
    mode = 1
    batch = 2 
    num_workers = 1
    SEED = 13    
    init_lr = 3e-4
    warmup_factor = 10 #how long
    warmup_epo = 1 
    log = True
    seed_everything(SEED)
    model = HUB_MODELS['efficientnet-b0']('efficientnet-b0')
    model.to(DEVICE)
    df = pd.read_csv(os.path.join(path_data, 'train_folds.csv'))
    
    kernel = type(model).__name__

    tr_idx = np.where(df.fold != fold)[0]
    vl_idx = np.where(df.fold == fold)[0]
    
    transforms_train = A.Compose([
        # A.OneOf([
        #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
        #     A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15),
        #     A.NoOp()        
        # ]),
        # A.OneOf([
        #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        #     A.RandomGamma(gamma_limit=(50, 150)),
        #     A.NoOp()        
        # ]),
        # A.OneOf([
        #     A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
        #     A.FancyPCA(3),
        #     A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5),
        #     A.NoOp()       
        # ]),
        # A.OneOf([
        #     A.CLAHE(),
        #     A.NoOp()         
        # ]),

        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5), 
        ]
    )
    
    # transforms_val = albumentations.Compose([])

    dataset = {
        'npy': [trainDataset_npy, 16],
        'pkl': [trainDataset_pkl, 25],
        'insta': [trainDataset_insta, None]        
        }


    trainDataset, num = dataset['pkl']
    
    td = trainDataset(df.iloc[tr_idx], df.iloc[tr_idx].isup_grade, num, rand = True, transform = transforms_train)    
    vd = trainDataset(df.iloc[vl_idx], df.iloc[vl_idx].isup_grade, num, rand = False, transform = transforms_train)
   
    train_dl = DataLoader(td, batch_size = batch,sampler=RandomSampler(td), num_workers=num_workers)
    val_dl = DataLoader(vd, batch_size = batch, sampler=SequentialSampler(vd), num_workers=num_workers)

    optimizer = Adam(model.parameters(), lr=init_lr/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    criterion = nn.BCEWithLogitsLoss()
    scaler = amp.GradScaler()
    qwk_max = 0
    for i in range(1, epoch + 1):
        print(f'Epoch: {i}')
        scheduler.step(i-1)
        model.train()            
        loss = train_epoch(model, train_dl, criterion, scaler, optimizer)  
        model.eval()
        with torch.no_grad():
            val_loss, pred, val_lab = train_epoch(model, val_dl, criterion, None, None)
        p = torch.cat(pred).cpu().numpy()
        t = torch.cat(val_lab).cpu().numpy()        
        acc = (p == t).mean() * 100.
        qwk = cohen_kappa_score(p, t, weights='quadratic')           
        #sch.step(val_loss)          #  Plateau            
        if log:
            print('Log.....')
            lg = time.ctime() + ' ' + f'Epoch {i}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(loss):.5f},  val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}, fold: {fold+1}'
            print(lg)
            with open(os.path.join(path_log, f'log_{kernel}_kaggle.txt'), 'a') as appender:
                appender.write(lg + '\n')            

        if qwk > qwk_max:
            print('Best ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
            torch.save(model.state_dict(), os.path.join(path_model, f'{kernel}_kaggle_best_fold{fold+1}_epoch_{i}.pth'))
            qwk_max = qwk
            
        #make checkpoint
        #problem in win
        # name_check = '_'.join(time.ctime().split(':')) + '_model.pt'

        # torch.save({
        #     'epoch': i,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()          
        #     }, os.path.join(path_checkpoint, name_check))        
            
    torch.save(model.state_dict(), os.path.join(path_model, '{kernel}_kaggle_final_fold{fold+1}.pth'))  

if __name__ == "__main__":
    main()


    
