import os
import time
import copy
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from models.utils import AverageMeter
from models.log import get_logger
from models.metric import IOUMetric
from models.losses import LovaszLoss, SoftCrossEntropyLoss, SoftBCEWithLogitsLoss
from models.dtd import seg_dtd
from models.tamper_dataset import TamperDataset
from swins import *
import torch.nn as nn

Image.MAX_IMAGE_PIXELS = 1000000000000


# def train_dtd(param, model, train_data_goodq, train_data_fairq, train_data_badq, train_data_all, valid_data, device='cuda'):
def train_dtd(param, model, train_data_goodq, valid_data, device='cuda'):
    # Initialize parameters
    model_name = param['model_name']
    epochs = param['epochs']
    batch_size = param['batch_size']
    iter_inter = param['iter_inter']
    save_log_dir = param['save_log_dir']
    save_ckpt_dir = param['save_ckpt_dir']
    load_ckpt_dir = param['load_ckpt_dir']
    save_epoch = param['save_epoch']
    T0 = param['T0']
    scaler = GradScaler()

    # Data loaders
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2, eta_min=1e-6, last_epoch=-1)
    
    # Loss functions
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    
    # Logger
    logger = get_logger(os.path.join(save_log_dir, time.strftime("%Y%m%d %H%M%S", time.localtime()) + '_' + model_name + '.log'))
    
    # Training loop
    best_iou = 0
    best_model = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info('Total Epoch:{} Training num:{} Validation num:{}'.format(epochs, len(train_data_q90), len(valid_data)))
    
    for epoch in range(epoch_start, epochs):
        epoch_start_time = time.time()
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        
        # Select training data based on epoch
        if epoch < 5:
            train_loader = DataLoader(dataset=train_data_goodq, batch_size=batch_size, shuffle=True, num_workers=1)
        # elif epoch < 10:
        #     train_loader = DataLoader(dataset=train_data_fairq, batch_size=batch_size, shuffle=True, num_workers=1)
        # elif epoch < 15:
        #     train_loader = DataLoader(dataset=train_data_badq, batch_size=batch_size, shuffle=True, num_workers=1)
        # else:
        #     train_loader = DataLoader(dataset=train_data_all, batch_size=batch_size, shuffle=True, num_workers=1)
        
        for batch_idx, batch_samples in enumerate(tqdm(train_loader)):
            data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'],batch_samples['rgb'], batch_samples['q'], batch_samples['i']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
            with autocast():
                pred = model(data, dct_coef, qs)
                # loss = LovaszLoss_fn(pred, target)
                lovasz_loss = LovaszLoss_fn(pred, target)
                bce_loss = SoftCrossEntropy_fn(pred, target)
                loss = lovasz_loss + bce_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / len(train_loader))
            train_epoch_loss.update(loss.item())
            train_iter_loss.update(loss.item())
            if batch_idx % iter_inter == 0:
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f}'.format(
                    epoch, batch_idx, len(train_loader), batch_idx / len(train_loader) * 100,
                    optimizer.param_groups[-1]['lr'], train_iter_loss.avg))
                train_iter_loss.reset()

        model.eval()
        valid_epoch_loss = AverageMeter()
        iou = IOUMetric(2)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tqdm(valid_loader)):
                data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'],batch_samples['rgb'], batch_samples['q'],batch_samples['i']
                data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
                pred = model(data, dct_coef, qs)
                lovasz_loss = LovaszLoss_fn(pred, target)
                bce_loss = SoftCrossEntropy_fn(pred, target)
                loss = lovasz_loss + bce_loss
                pred = pred.cpu().data.numpy()
                pred = np.argmax(pred, axis=1)
                iou.add_batch(pred, target.cpu().data.numpy())
                valid_epoch_loss.update(loss.item())
        
        acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
        logger.info('[val] epoch:{} iou:{}'.format(epoch, iu))

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info('[time] epoch:{} duration:{:.2f}s'.format(epoch, epoch_duration))
        
        if iu[1] > best_iou:
            best_iou = iu[1]
            best_model = copy.deepcopy(model)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(save_ckpt_dir, 'checkpoint-best.pth'))
            logger.info('[save] Best Model saved at epoch:{}'.format(epoch))
        
        if epoch in save_epoch[T0]:
            torch.save(model.state_dict(), os.path.join(save_ckpt_dir, 'cosine_epoch{}.pth'.format(epoch)))
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(save_ckpt_dir, 'checkpoint-latest.pth'))
    
    return best_model, model


if __name__ == "__main__":


    # Parameters
    param = {
        'epochs': 1,      # number of epochs
        'batch_size': 4, # batch size
        'iter_inter': 50, # print loss info every iter_inter iterations
        'model_name': 'DTD',  # DocTamperDetection
        'save_log_dir': './outputs/logs',
        'save_ckpt_dir': './outputs/ckpt',
        'T0': 2,          # CosineAnnealingWarmRestarts T_0
        'save_epoch': {2: [5, 13, 29, 61], 3: [8, 20, 44, 92]},  # save model at epoch# {T0: [epochs]}
        'load_ckpt_dir': None
    }

    # Create directories if not exist
    if not os.path.exists(param['save_ckpt_dir']):
        os.makedirs(param['save_ckpt_dir'])
    if not os.path.exists(param['save_log_dir']):
        os.makedirs(param['save_log_dir'])

    # Load data
    train_data_q90 = TamperDataset(r'../dataset/DocTamperV1-SCD',True,minq=90)
    # train_data_q85 = TamperDataset(r'../dataset/DocTamperV1-TestingSet',True,minq=85)
    # train_data_q80 = TamperDataset(r'../dataset/DocTamperV1-TestingSet',True,minq=80)
    # train_data_q75 = TamperDataset(r'../dataset/DocTamperV1-TestingSet',True,minq=75)
    valid_data = TamperDataset(r'../dataset/DocTamperV1-SCD',True,minq=75)
    # test_data = TamperDataset(args.data_root+args.lmdb_name,False,minq=args.minq)

    # Initialize model
    # model = DTD().cuda()
    model = seg_dtd('', 2).cuda()
    model = torch.nn.DataParallel(model)

    ckpt = torch.load('weights/dtd_doctamper.pth')
    # model.load_state_dict(ckpt['state_dict'])

    # Train model
    # best_model, model = train_dtd(param, model, train_data_q90, train_data_q85, train_data_q80, train_data_q75, valid_data)
    best_model, model = train_dtd(param, model, train_data_q90, valid_data)
