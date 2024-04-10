import argparse

parser = argparse.ArgumentParser(description='SODR Pre-Training Stage')

# Hardware
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--pin_memory', action='store_true')

# Train Common
parser.add_argument('--exp_name', type=str)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoches', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.06)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--backbone', type=str, default='resnet18')
# parser.add_argument('--hid_dim', type=int, default=2048)

# Resume
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--model_load_path', type=str)

args, unparsed = parser.parse_known_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False