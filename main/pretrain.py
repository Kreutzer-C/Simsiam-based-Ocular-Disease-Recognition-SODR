import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import sys, os

sys.path.append(os.getcwd())
sys.path.append(r'D:\SODR') # Change it to your own

from pt_args import args
from tools.save_tools import save_to_file, save_args2file
from dataset.augmentations import *
from dataset.MyDataset import ODIRDataSet_Sim
from model.simsiam import SimSiam

# Preparing save path
log_save_path = f"../result/pretrain/log/{args.exp_name}"
loss_save_path = f"../result/pretrain/loss/{args.exp_name}"

all_save_path = [log_save_path, loss_save_path]
for path in all_save_path:
    if not os.path.exists(path):
        os.makedirs(path)

# Save the run command
p_str = "".join(sys.argv)
save_to_file(os.path.join(log_save_path, 'pretrain_log.txt'), p_str + '\n')

print("\n>>>=====args=====<<<\n")
save_args2file(os.path.join(log_save_path, 'pretrain_log.txt'), args)
print("\n>>>==============<<<\n")

device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print("\n>>>=====device:{}({},{}MB=====<<<)\n".format(device, GPU_device.name, GPU_device.total_memory / 1024 ** 2))

# Load dataset
augmentation = SimSiamTransform()
train_dataset = ODIRDataSet_Sim("../dataset/ODIR-5K_Training_Dataset", transform=augmentation)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)

# Define Model
model = SimSiam(args).to(device, non_blocking=True)

# Define Loss function & Optimizer
criterion = nn.CosineSimilarity(dim=1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Load the model(resume from checkpoint)
if args.start_epoch != 0:
    checkpoint = torch.load(os.path.join(args.model_load_path, 'simsiam.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_history = checkpoint['loss_history']

# Begin Training
print("\n>>>=====Start Training=====<<<\n")
if args.start_epoch == 0:
    loss_history = {'loss_tr': []}

for epoch in tqdm(range(args.start_epoch, args.epoches)):
    model.train()
    total_loss = 0
    for iteration, (x1, x2) in enumerate(train_loader):
        if (iteration % 10) == 0:
            print(f"Training>=={iteration}/{len(train_loader)}<==")
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad()
        p1, p2, z1, z2 = model(x1, x2)

        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        total_loss += loss.item()
        loss.backward()

        optimizer.step()

    avg_loss = total_loss/len(train_loader)
    loss_history['loss_tr'].append(avg_loss)

    print_log = "epoch:{} ==Train[Loss:{:.5f}]".format(epoch, avg_loss)
    print(print_log)
    save_to_file(os.path.join(log_save_path, 'pretrain_log.txt'), print_log)

    pth_save_path = f"../result/pretrain/checkpoint/{args.exp_name}/epoch{epoch}"
    if not os.path.exists(pth_save_path):
        os.makedirs(pth_save_path)
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss_history':loss_history},
               os.path.join(pth_save_path, 'simsiam.pth'))
    torch.save({'model_state_dict':model.encoder.state_dict()},
               os.path.join(pth_save_path, 'encoder.pth'))

    df = pd.DataFrame(loss_history)
    df.to_csv(os.path.join(loss_save_path, 'loss.csv'), index=False)
