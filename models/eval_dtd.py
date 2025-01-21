from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from models.metric import IOUMetric
from models.dtd import seg_dtd
from models.tamper_dataset import TamperDataset
from swins import *
import torch.nn as nn
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_root', type=str, default='../dataset/') # root to the dir of lmdb files
# parser.add_argument('--pth', type=str, default='weights/dtd_doctamper.pth')
# parser.add_argument('--lmdb_name', type=str, default='DocTamperV1-SCD')
# parser.add_argument('--minq', type=int, default=75)
# args = parser.parse_args()



def eval_net_dtd(model, test_data, plot=False,device='cuda'):
    eval_loader = DataLoader(dataset=test_data, batch_size=6, num_workers=12, shuffle=False)

    model.eval()
    iou=IOUMetric(2)
    precisons = []
    recalls = []
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(eval_loader)):
            data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'],batch_samples['rgb'], batch_samples['q'],batch_samples['i']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
            pred = model(data,dct_coef,qs)
            predt = pred.argmax(1)
            pred=pred.cpu().data.numpy()
            targt = target.squeeze(1)
            matched = (predt*targt).sum((1,2))
            pred_sum = predt.sum((1,2))
            target_sum = targt.sum((1,2))
            precisons.append((matched/(pred_sum+1e-8)).mean().item())
            recalls.append((matched/target_sum).mean().item())
            pred = np.argmax(pred,axis=1)
            iou.add_batch(pred,target.cpu().data.numpy())
        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu,precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8))))

if __name__ == "__main__":
    # test_data = TamperDataset(args.data_root+args.lmdb_name,False,minq=args.minq)
    test_data = TamperDataset(r'../dataset/DocTamperV1-SCD', True, minq=75)

    model = seg_dtd('', 2).cuda()
    model = torch.nn.DataParallel(model)

    ckpt = torch.load('weights/dtd_doctamper.pth')
    model.load_state_dict(ckpt['state_dict'])

    eval_net_dtd(model, test_data)


