import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os,sys

sys.path.append(os.getcwd())
sys.path.append(r'D:\SODR') # Change it to your own

from ft_args import args
from tools.save_tools import save_to_file, save_args2file
from dataset.augmentations import *
from dataset.MyDataset import ODIRDataSet_ft
from model.ft_model import FTmodel
from tools.loss_weight import cal_loss_weight
from tools.metrics import calculate_metrics, show_metrics

import warnings
warnings.filterwarnings("ignore")

# Preparing save path
pth_best_save_path = f"../result/finetune/checkpoint/{args.exp_name}/best"
log_save_path = f"../result/finetune/log/{args.exp_name}"
loss_save_path = f"../result/finetune/loss/{args.exp_name}"

all_save_path = [pth_best_save_path, log_save_path, loss_save_path]
for path in all_save_path:
    if not os.path.exists(path):
        os.makedirs(path)

# Save the run command
p_str = "".join(sys.argv)
save_to_file(os.path.join(log_save_path, 'finetune_log.txt'), p_str + '\n')

print("\n>>>=====args=====<<<\n")
save_args2file(os.path.join(log_save_path, 'finetune_log.txt'), args)
print("\n>>>==============<<<\n")

device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print("\n>>>=====device:{}({},{}MB=====<<<)\n".format(device, GPU_device.name, GPU_device.total_memory / 1024 ** 2))

# Read label file
csv_path = "../dataset/ODIR-5K_Annotations.csv"
df = pd.read_csv(csv_path)

# Split train, val and test
train_df, val_df = train_test_split(df, test_size=0.1)

# Load dataset
augmentation_tr = Transform_single()
augmentation_ts = Transform_test()
train_dataset = ODIRDataSet_ft(train_df, transform=augmentation_tr)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)
val_dataset = ODIRDataSet_ft(val_df, transform=augmentation_ts)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)

# Define Model
model = FTmodel(args).to(device, non_blocking=True)

# Define Loss function & Optimizer
if args.scratch:
    criterion = nn.BCEWithLogitsLoss()
else:
    loss_weight = cal_loss_weight(df)
    loss_weight = torch.tensor(loss_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(loss_weight)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

# Load the model(from pre-train or resume checkpoint)
if args.start_epoch == 0: # From pre-train
    if not args.scratch:
        checkpoint = torch.load(os.path.join(args.sim_load_path, 'encoder.pth'))
        model.encoder.load_state_dict(checkpoint['model_state_dict'])
else: # Fine-tune resume
    checkpoint = torch.load(os.path.join(args.model_load_path, 'ftmodel.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    f1_best = checkpoint['f1_best']
    loss_history = checkpoint['loss_history']

print("\n>>>=====Start Training=====<<<\n")
if args.start_epoch == 0:
    loss_history = {'loss_tr': [], 'loss_val':[]}
    f1_best = 0

for epoch in tqdm(range(args.start_epoch, args.epoches)):
    model.train()
    total_loss_tr = 0

    for iteration, (img1, img2, labels) in enumerate(train_loader):
        if (iteration % 10) == 0:
            print(f"Training>=={iteration}/{len(train_loader)}<==")
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        output1, output2 = model(img1, img2)

        loss_l = criterion(output1, labels)
        loss_r = criterion(output2, labels)
        loss = loss_l + loss_r
        total_loss_tr += loss.item()
        loss.backward()
        optimizer.step()

    average_loss_tr = total_loss_tr / len(train_loader)

    # Validation
    model.eval()
    total_loss_val = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for iteration, (img1, img2, labels) in enumerate(val_loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output1, output2 = model(img1, img2)

            loss_l = criterion(output1, labels)
            loss_r = criterion(output2, labels)
            loss = loss_l + loss_r
            total_loss_val += loss.item()

            predictions = (output1.sigmoid() + output2.sigmoid()) / 2.0
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    average_loss_val = total_loss_val / len(val_loader)
    loss_history['loss_tr'].append(average_loss_tr)
    loss_history['loss_val'].append(average_loss_val)
    print("epoch:{} ===[Train Loss:{:.5f} Val Loss:{:.5f}]".format(epoch, average_loss_tr, average_loss_val))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc, precision, recall, f1, num = calculate_metrics(all_predictions, all_labels, args.threshold)
    average_acc, average_f1 = show_metrics(acc, precision, recall, f1, num)
    print_log = "epoch:{} ===[Train Loss:{:.5f} Val Loss:{:.5f}][Acc:{:.5f} F1:{:.5f}]".format(epoch,
                                                                                               average_loss_tr,
                                                                                               average_loss_val,
                                                                                               average_acc,
                                                                                               average_f1)
    save_to_file(os.path.join(log_save_path, 'finetune_log.txt'), print_log)

    if average_f1 >= f1_best:
        f1_best = average_f1
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'f1_best':f1_best,
                    'loss_history':loss_history},
                   os.path.join(pth_best_save_path, 'ftmodel.pth'))
        log = f"A new record has been saved."
        print(log)
        save_to_file(os.path.join(log_save_path, 'finetune_log.txt'),log)

    pth_save_path = f"../result/finetune/checkpoint/{args.exp_name}/epoch{epoch}"
    if not os.path.exists(pth_save_path):
        os.makedirs(pth_save_path)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_best': f1_best,
                'loss_history': loss_history},
               os.path.join(pth_save_path, 'ftmodel.pth'))

    df = pd.DataFrame(loss_history)
    df.to_csv(os.path.join(loss_save_path, 'loss.csv'), index=False)
