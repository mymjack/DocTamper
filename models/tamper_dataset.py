import os.path

import cv2
import lmdb
import jpegio
import pickle
import six
from PIL import Image
from torch.utils.data import Dataset
from dtd import *
from albumentations.pytorch import ToTensorV2
import torchvision
import tempfile
import numpy as np

def get_lmdb_size(roots, max_readers=64):
    envs = lmdb.open(roots,max_readers=max_readers,readonly=True,lock=False,readahead=False,meminit=False)
    with envs.begin(write=False) as txn:
        size = int(txn.get('num-samples'.encode('utf-8')))
    envs.close()
    envs = None
    return size

class TamperDataset(Dataset):
    def __init__(self, roots, mode, minq=95, qtb=90, max_readers=64):
        self.roots = roots
        self.max_readers = max_readers
        self.envs = None
        self.max_nums = self.nSamples = get_lmdb_size(roots, max_readers)
        self.minq = minq
        self.mode = mode
        self.pks = {}
        with open('../qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        for k,v in pks.items():
            self.pks[k] = torch.LongTensor(v)
        with open('../pks/'+os.path.basename(roots)+'_%d.pk'%minq,'rb') as f:
            self.record = pickle.load(f)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = ToTensorV2()
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])
        
    # def _init_db(self):
    #     self.envs = lmdb.open(self.roots,max_readers=self.max_readers,readonly=True,lock=False,readahead=False,meminit=False)
    #     with self.envs.begin(write=False) as txn:
    #         self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
    #     self.max_nums=self.nSamples

    def __len__(self):
        return self.max_nums

    def __getitem__(self, index):
        if self.envs is None:
            # self._init_db()
            self.envs = lmdb.open(self.roots,max_readers=self.max_readers,readonly=True,lock=False,readahead=False,meminit=False)
        with self.envs.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            im = Image.open(buf)
            lbl_key = 'label-%09d' % index
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = (cv2.imdecode(np.frombuffer(lblbuf,dtype=np.uint8),0)!=0).astype(np.uint8)
            H,W = mask.shape
            record = self.record[index]
            choicei = len(record)-1
            q = int(record[-1])
            use_qtb = self.pks[q]
            if choicei>1:
                q2 = int(record[-3])
                use_qtb2 = self.pks[q2]
            if choicei>0:
                q1 = int(record[-2])
                use_qtb1 = self.pks[q1]
            mask = self.totsr(image=mask.copy())['image']
            # with tempfile.NamedTemporaryFile(delete=True) as tmp:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_filename = tmp.name
            try:
                im = im.convert("L")
                if choicei>1:
                    im.save(tmp_filename,"JPEG",quality=q2)
                    im = Image.open(tmp_filename)
                if choicei>0:
                    im.save(tmp_filename,"JPEG",quality=q1)
                    im = Image.open(tmp_filename)
                im.save(tmp_filename,"JPEG",quality=q)
                jpg = jpegio.read(tmp_filename)
                dct = jpg.coef_arrays[0].copy()
                im = im.convert('RGB')
            finally:
                os.remove(tmp_filename)
            return {
                'image': self.toctsr(im),
                'label': mask.long(),
                'rgb': np.clip(np.abs(dct),0,20),
                'q':use_qtb,
                'i':q
            }