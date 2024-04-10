import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np
import os,sys

sys.path.append(os.getcwd())
sys.path.append(r'D:\SODR') # Change it to your own

from ft_args import args
from tools.save_tools import save_to_file, save_args2file
from dataset.augmentations import *
from dataset.MyDataset import ODIRDataSet_oltest
from model.ft_model import FTmodel

import warnings
warnings.filterwarnings("ignore")

# Preparing save path
log_save_path = f"../result/test_ol/log/{args.exp_name}"
submit_save_path = f"../result/test_ol/submit/{args.exp_name}"
if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)
if not os.path.exists(submit_save_path):
    os.makedirs(submit_save_path)

# Save the run command
p_str = "".join(sys.argv)
save_to_file(os.path.join(log_save_path, 'testol_log.txt'), p_str + '\n')

print("\n>>>=====args=====<<<\n")
save_args2file(os.path.join(log_save_path, 'testol_log.txt'), args)
print("\n>>>==============<<<\n")

device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print("\n>>>=====device:{}({},{}MB=====<<<)\n".format(device, GPU_device.name, GPU_device.total_memory / 1024 ** 2))

# Load dataset
augmentation_ts = Transform_test()
test_dataset = ODIRDataSet_oltest("../dataset/ODIR-5K_Testing_Images", transform=augmentation_ts)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)

# Define & load Model
model = FTmodel(args).to(device, non_blocking=True)
checkpoint = torch.load(os.path.join(args.model_load_path, 'ftmodel.pth'))
model.load_state_dict(checkpoint['model_state_dict'])

# Test
print("\n>>>=====Start Testing=====<<<\n")
model.eval()
with torch.no_grad():
    all_id = []
    all_predictions = []
    for iteration, (id, img1, img2) in enumerate(test_loader):
        # if iteration % (len(test_loader)/5) == 0:
        #     print(f"===Testing {iteration}/{len(test_loader)}")
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        output1, output2 = model(img1, img2)

        predictions = (output1.sigmoid() + output2.sigmoid()) / 2.0
        all_id.append(id)
        all_predictions.append(predictions.cpu().numpy())

    all_id = np.concatenate(all_id, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    print(all_id.shape, all_predictions.shape)
    df = pd.DataFrame(all_predictions, columns=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    df.insert(0, 'ID', all_id)

    df.to_csv(os.path.join(submit_save_path, 'IAIR_ODIR.csv'), index=False)


