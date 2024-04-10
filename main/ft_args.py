import argparse

parser = argparse.ArgumentParser(description='SODR Fine-Tuning Stage')

# Hardware
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--pin_memory', action='store_true')

# Train Common
parser.add_argument('--exp_name', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoches', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--num_class', type=int, default=8)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--sim_load_path', type=str)
parser.add_argument('--forzen_en', action='store_true')
parser.add_argument('--scratch', action='store_true')

# Resume
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--model_load_path', type=str)

args, unparsed = parser.parse_known_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False