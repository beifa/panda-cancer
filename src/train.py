import os
import torch
import time
import random
import albumentations
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import torch.nn as nn
from dataset import trainDataset_insta, trainDataset_pkl, trainDataset_npy
from hub_models import HUB_MODELS
from torch.optim import Adam, SGD
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler 
import joblib
#print(torch.__version__)
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


def train_epoch(model, loader, optimizer, l_fnc, scaler):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        #with amp.autocast():
        y_ = model(data)
        loss = l_fnc(y_, target)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        optimizer.step()
        
        l_np = loss.detach().cpu().numpy()
        train_loss.append(l_np)
    return train_loss

def val_epoch(model, dataloader, l_fnc):
    losses =[]
    preds = []
    labels = []
    acc = 0.
    with torch.no_grad():
        for (data, target) in tqdm(dataloader): 
            data, target = data.to(DEVICE), target.to(DEVICE) 
            y_ = model(data)
            loss = l_fnc(y_, target)
            pred = y_.sigmoid().sum(1).detach().round()
            #pred = y_.sum(1).detach().round()
            preds.append(pred)
            losses.append(loss.detach().cpu().numpy())
            labels.append(target.sum(1))
        losses = np.mean(losses)

    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    acc = (preds == labels).mean() * 100.

    qwk = cohen_kappa_score(preds, labels, weights='quadratic')
    return losses, acc, qwk


def main():
    fold = 0
    epoch = 30   
    mode = 1
    batch = 2 #3 
    num_workers = 3
    SEED = 13
    log = True
    seed_everything(SEED)
    model = HUB_MODELS['efficientnet-b0']('efficientnet-b0')
    #model.load_state_dict(torch.load(os.path.join(path_model, 'Net_kaggle_final_fold1.pth')))
    model.to(DEVICE)
    df = pd.read_csv(os.path.join(path_data, 'train_folds.csv'))

    kernel = type(model).__name__

    tr_idx = np.where(df.fold != fold)[0]
    vl_idx = np.where(df.fold == fold)[0]

    transforms_train = albumentations.Compose([
                                            albumentations.Transpose(p=0.5),
                                            albumentations.VerticalFlip(p=0.5),
                                            albumentations.HorizontalFlip(p=0.5),
                                            ])
    
    # transforms_val = albumentations.Compose([])

    dataset = {
        'npy': [trainDataset_npy, 16],
        'pkl': [trainDataset_pkl, 25],
        'insta': [trainDataset_insta, None]        
        }


    trainDataset, num = dataset['pkl']
        
    td = trainDataset(df.iloc[tr_idx], df.iloc[tr_idx].isup_grade, num, rand = True, transform = transforms_train)
    
    vd = trainDataset(df.iloc[vl_idx], df.iloc[vl_idx].isup_grade, num, rand = False, transform =None)
    
    train_dl = DataLoader(td, batch_size = batch,sampler=RandomSampler(td), num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(vd, batch_size = batch, sampler=SequentialSampler(vd), num_workers=num_workers)

    init_lr = 0.0000035 #3e-4
    warmup_factor = 10 #how long
    warmup_epo = 1 

    optimizer = Adam(model.parameters(), lr=init_lr/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    criterion = nn.BCEWithLogitsLoss()

    scaler = amp.GradScaler() #torch17

    qwk_max = 0
    for i in range(1, epoch + 1):
        print(f'Epoch: {i}')
        scheduler.step(i-1)
        loss = train_epoch(model, train_dl, optimizer, criterion, scaler)
        val_loss, acc, qwk = val_epoch(model, val_dl, criterion)            
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


    #clear cash
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


    
