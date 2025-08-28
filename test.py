from models.net import Encoder, Decoder, iv_fuse_sche
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import models as Model
import argparse
import logging
import utils.logger as Logger
import cv2
from models.sr3_modules.unet import CBAM

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/diffusion.json', help='configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='testing', default='test')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
args = parser.parse_args()
opt = Logger.parse(args)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = r"weights/dmnet.pth"
Encoder = nn.DataParallel(Encoder()).to(device)
Decoder = nn.DataParallel(Decoder()).to(device)
Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
CBAM = CBAM(in_channels=64).to(device)

# for dataset_name in ["TNO", "RoadScene","MSRS"]:
for dataset_name in ["MSRS"]:
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('test_result',dataset_name)
    Encoder.eval()
    Decoder.eval()
    diffusion = Model.create_model(opt)
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):
            print(img_name)
            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))
            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            feature_V_D, v64t1, v64t2, v64t3, v64t4, v64t5 = Encoder(data_VIS)
            feature_I_D, i64t1, i64t2, i64t3, i64t4, i64t5 = Encoder(data_IR)
            at_feature_V_D = CBAM(feature_V_D)
            at_feature_I_D = CBAM(feature_I_D)
            data_fuse = iv_fuse_sche(sche='sum', f_v=at_feature_V_D, f_i=at_feature_I_D)
            reduce_channel1 = nn.Conv2d(64, 1, kernel_size=1, bias=False).to(device)
            vt1, vt2, vt3, vt4, vt5 = reduce_channel1(v64t1), reduce_channel1(v64t2), reduce_channel1(v64t3), reduce_channel1(v64t4), reduce_channel1(v64t5)
            it1, it2, it3, it4, it5 = reduce_channel1(i64t1), reduce_channel1(i64t2), reduce_channel1(i64t3), reduce_channel1(i64t4), reduce_channel1(i64t5)
            diff_fuse0 = torch.cat((data_VIS, data_IR), dim=1)
            diff_fuse1 = torch.cat((vt1, it1), dim=1)
            diff_fuse2 = torch.cat((vt2, it2), dim=1)
            diff_fuse3 = torch.cat((vt3, it3), dim=1)
            diff_fuse4 = torch.cat((vt4, it4), dim=1)
            diff_fuse5 = torch.cat((vt5, it5), dim=1)
            diff_fuse_avg = (diff_fuse0 + diff_fuse1 + diff_fuse2 + diff_fuse3 + diff_fuse4 + diff_fuse5) / 6
            diffusion.feed_data(diff_fuse_avg)
            av = torch.zeros(len(diff_fuse0), 64, len(diff_fuse0[0][0]), len(diff_fuse0[0][0][0])).to('cuda')
            for t in opt['model_df']['t']:
                fe_t, fd_t = diffusion.get_feats(t=t)
                av += fd_t
            av /= len(opt['model_df']['t'])
            data_fuse = data_fuse + av
            data_Fuse = Decoder(data_VIS, data_fuse)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255.0).cpu().numpy())
            fi = fi.astype(np.uint8)
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
