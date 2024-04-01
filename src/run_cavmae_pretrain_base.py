import argparse
import os
import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

#### ---------> check if work on clip
# import dataloader_clip as dataloader
# import dataloader_clip_val as dataloader_val

import dataloader as dataloader
import dataloader_val as dataloader_val
#### <-------
import models
import numpy as np
from traintest_cavmae_base import train
import wandb
import utils
import random

from torch.nn.parallel import DistributedDataParallel as DDP
from seq_dataloader import SequentialDistributedSampler


def init_seeds(seed=0, cuda_deterministic=True):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	# Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
	if cuda_deterministic:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

# pretrain cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400", "msrvtt"])
parser.add_argument("--dataset_mean", type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num_workers', default=6, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments, only for preliminary experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--n-print-steps", type=int, default=50, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--cont_model", help='previous pretrained model', type=str, default=None)
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=None)
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default=None)
parser.add_argument("--masking_ratio", type=float, default=0.75, help="masking ratio")
parser.add_argument("--masking_ratio_a", type=float, default=None, help="masking ratio")


parser.add_argument("--mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])


# wandb
parser.add_argument("--wandb", type=int, default=0, help="wandb")
parser.add_argument('--model_name', type=str, default=None, help="for log")


# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_url', default='env://',
					help='url used to set up distributed training')


args = parser.parse_args()

if args.masking_ratio_a is None:
    args.masking_ratio_a = args.masking_ratio

args.local_rank = int(os.environ["LOCAL_RANK"]) # for torchrun
init_seeds(87 + args.local_rank, cuda_deterministic=False) #: add
utils.init_distributed_mode(args)

if args.wandb:
	if args.rank == 0:
		wandb.init(config=args, project="uavm", name=args.model_name)

im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
			  'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
				  'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

if args.bal == 'bal':
	print('balanced sampler is being used')
	if args.weight_file == None:
		samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
	else:
		samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
	sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

	train_loader = torch.utils.data.DataLoader(
		dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
		batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
else:
	print('balanced sampler is not used')

	train_sampler = torch.utils.data.distributed.DistributedSampler(dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),shuffle=True)
	# train_loader = torch.utils.data.DataLoader(
	# 	dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
	# 	batch_size=args.batch_size, sampler=train_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

	train_sampler_linear = torch.utils.data.distributed.DistributedSampler(dataloader_val.AudiosetDataset('/mnt/opr/yblin/audioset_sun/audioset_20k_cleaned.json', label_csv=args.label_csv, audio_conf=audio_conf),shuffle=True)
	train_loader_linear = torch.utils.data.DataLoader(
		dataloader_val.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
		batch_size=16, sampler=train_sampler_linear, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

# val_loader = torch.utils.data.DataLoader(
#     dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
#     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

val_sampler = SequentialDistributedSampler(dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=args.batch_size)
val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), 
batch_size=args.batch_size, num_workers=args.num_workers, sampler=val_sampler, pin_memory=False, persistent_workers=False)


val_sampler_linear = SequentialDistributedSampler(dataloader_val.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=args.batch_size)
val_loader_linear = torch.utils.data.DataLoader(dataloader_val.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), 
batch_size=args.batch_size, num_workers=args.num_workers, sampler=val_sampler_linear, pin_memory=False, persistent_workers=False)


if args.data_eval != None:
	eval_loader = torch.utils.data.DataLoader(
		dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
		batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

if args.model == 'cav-mae':
	print('pretrain a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')

	### --------> check if clip version 
	audio_model = models.CAVMAE_BASE(audio_length=args.target_length, norm_pix_loss=args.norm_pix_loss, modality_specific_depth=23, tr_pos=args.tr_pos, opt=args)
	# audio_model = models.CAVMAE_BASE_CLIP(audio_length=args.target_length, norm_pix_loss=args.norm_pix_loss, modality_specific_depth=23, tr_pos=args.tr_pos, opt=args)
	# audio_model = models.CAVMAE_BASE_DINO(audio_length=args.target_length, norm_pix_loss=args.norm_pix_loss, modality_specific_depth=23, tr_pos=args.tr_pos, opt=args)
else:
	raise ValueError('model not supported')

# initialized with a pretrained checkpoint (e.g., original vision-MAE checkpoint)
# if args.pretrain_path != 'None':
# 	mdl_weight = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
# 	# if not isinstance(audio_model, torch.nn.DataParallel):
# 	#     audio_model = torch.nn.DataParallel(audio_model)
# 	miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
# 	print('now load mae pretrained weights from ', args.pretrain_path)
# 	print(miss, unexpected)

	# audio_model = audio_model.module.to(torch.device('cpu'))

# if args.cont_model != None:
#     print('now load pretrained weights from : ' + args.cont_model)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sdA = torch.load(args.cont_model, map_location=device)
#     if isinstance(audio_model, torch.nn.DataParallel) == False:
#         audio_model = torch.nn.DataParallel(audio_model)
#     audio_model.load_state_dict(sdA, strict=True)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
	os.makedirs("%s/models" % args.exp_dir)
except:
	pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
	pickle.dump(args, f)
with open(args.exp_dir + '/args.json', 'w') as f:
	json.dump(args.__dict__, f, indent=2)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
# train(audio_model, train_loader, [val_loader, val_loader_linear], [val_sampler, val_sampler_linear], train_loader_linear, args)
train(audio_model, train_sampler, [val_loader, val_loader_linear], [val_sampler, val_sampler_linear], train_loader_linear, args, audio_conf)

# train(audio_model, train_loader, val_loader, val_loader, args)