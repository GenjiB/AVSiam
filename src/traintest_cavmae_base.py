import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import dataloader as dataloader
from torch.nn.parallel import DistributedDataParallel as DDP

from ipdb import set_trace
import wandb
import models

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

def distributed_concat(tensor, num_total_examples):
	output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(output_tensors, tensor)
	concat = torch.cat(output_tensors, dim=0)
	# truncate the dummy elements added by SequentialDistributedSampler
	return concat[:num_total_examples]

# def train(audio_model, train_loader, test_loader, test_sampler, train_loader_linear, args):
def train(audio_model, train_sampler, test_loader, test_sampler, train_loader_linear, args, audio_conf):

	test_loader, test_loader_linear =test_loader[0], test_loader[1]
	test_sampler, test_sampler_linear = test_sampler[0], test_sampler[1]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('running on ' + str(device))
	torch.set_grad_enabled(True)

	batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
	loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
	progress = []

	best_epoch, best_loss = 0, np.inf
	global_step, epoch = 0, 0
	start_time = time.time()
	exp_dir = args.exp_dir

	def _save_progress():
		progress.append([epoch, global_step, best_epoch, best_loss,
				time.time() - start_time])
		with open("%s/progress.pkl" % exp_dir, "wb") as f:
			pickle.dump(progress, f)

	# if not isinstance(audio_model, nn.DataParallel):
	#     audio_model = nn.DataParallel(audio_model)
	# audio_model = audio_model.to(device)
	
	device = torch.device("cuda", args.local_rank)
	audio_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(audio_model).to(device)
	audio_model = DDP(audio_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)    

	trainables = [p for p in audio_model.parameters() if p.requires_grad]
	print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
	print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
	optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

	optimizer2 = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

	# use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
	if args.lr_adapt == True:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
		print('Override to use adaptive learning rate scheduler.')
	else:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
		scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
		print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

	print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

	# #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
	# torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

	epoch += 1
	scaler = GradScaler()
	scaler2 = GradScaler()

	print("current #steps=%s, #epochs=%s" % (global_step, epoch))
	print("start training...")
	result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
	audio_model.train()
	


	train_loader = torch.utils.data.DataLoader(
	dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
	batch_size=args.batch_size, sampler=train_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)


	while epoch < args.n_epochs + 1:
		begin_time = time.time()
		end_time = time.time()
		audio_model.train()
		print('---------------')
		print(datetime.datetime.now())
		print("current #epochs=%s, #steps=%s" % (epoch, global_step))
		print('current masking ratio is {:.3f} for audio {:.3f} for video; audio mask mode {:s}'.format(args.masking_ratio_a,args.masking_ratio, args.mask_mode))
		# print('current masking ratio is {:.3f} for audio {:.3f} for video; audio mask mode {:s}'.format(masking[epoch-1],masking[epoch-1], args.mask_mode))





		for i, (a_input, v_input, _) in enumerate(train_loader):

			B = a_input.size(0)
			a_input = a_input.to(device, non_blocking=True)
			v_input = v_input.to(device, non_blocking=True)

			data_time.update(time.time() - end_time)
			per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
			dnn_start_time = time.time()
			


			# ########### -------> Flops
			# prof = FlopsProfiler(audio_model)
			# prof.start_profile()
			# start = time.process_time()

			# ########## <-------

			with autocast():
				loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(a_input, v_input, args.masking_ratio_a, args.masking_ratio, mae_loss_weight=0, contrast_loss_weight=1, mask_mode=args.mask_mode)




			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			




			with autocast():
				loss, loss_mae, loss_mae_a, loss_mae_v, _, mask_a, mask_v, _ = audio_model(a_input, v_input, args.masking_ratio_a, args.masking_ratio, mae_loss_weight=1, contrast_loss_weight=0, mask_mode=args.mask_mode)

			optimizer2.zero_grad()
			scaler2.scale(loss).backward()
			scaler2.step(optimizer2)
			scaler2.update()
			
			

			# scaler.step(optimizer)
			# scaler.update()

			# loss_av is the main loss
			loss_av_meter.update(loss.item(), B)
			loss_a_meter.update(loss_mae_a.item(), B)
			loss_v_meter.update(loss_mae_v.item(), B)
			loss_c_meter.update(loss_c.item(), B)
			batch_time.update(time.time() - end_time)
			per_sample_time.update((time.time() - end_time)/a_input.shape[0])
			per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

			print_step = global_step % args.n_print_steps == 0
			early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
			print_step = print_step or early_print_step

			if print_step or global_step == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
				  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
				  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
				  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
				  'Train Total Loss {loss_av_meter.val:.4f}\t'
				  'Train MAE Loss Audio {loss_a_meter.val:.4f}\t'
				  'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
				  'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
				  'Train Contrastive Acc {c_acc:.3f}\t'.format(
				   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
					  per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
				if np.isnan(loss_av_meter.avg):
					print("training diverged...")
					return

			end_time = time.time()
			global_step += 1

		print('start validation')
		eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(audio_model, test_loader, test_sampler, args)

		print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
		print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
		print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
		print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
		print("Eval Total Loss: {:.6f}".format(eval_loss_av))
		print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

		print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
		print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
		print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
		print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))

		# train audio mae loss, train visual mae loss, train contrastive loss, train total loss
		# eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
		result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
		np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
		print('validation finished')


		if args.wandb:
			if args.rank == 0:
				wandb.log({"Eval Audio MAE Loss": eval_loss_mae_a})
				wandb.log({"Eval Visual MAE Loss": eval_loss_mae_v})
				wandb.log({"Eval Total MAE Loss": eval_loss_mae})
				wandb.log({"Eval Contrastive Loss": eval_loss_c})
				wandb.log({"Eval Total Loss": eval_loss_av})
				wandb.log({"Eval Contrastive Accuracy": eval_c_acc})
	

		if eval_loss_av < best_loss:
			best_loss = eval_loss_av
			best_epoch = epoch

		if best_epoch == epoch:
			if args.rank == 0:
				torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
				torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

		if args.save_model == True:
			if args.rank == 0:
				torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

		if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
			scheduler.step(-eval_loss_av)
		else:
			scheduler.step()
			scheduler2.step()

		print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

		_save_progress()

		finish_time = time.time()
		print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))


		print("######### Running MLP tuning  ################")
		torch.distributed.barrier()
		linear_val(train_loader_linear, test_loader_linear, test_sampler_linear, epoch=epoch,args=args)


		epoch += 1
		batch_time.reset()
		per_sample_time.reset()
		data_time.reset()
		per_sample_data_time.reset()
		per_sample_dnn_time.reset()
		loss_av_meter.reset()
		loss_a_meter.reset()
		loss_v_meter.reset()
		loss_c_meter.reset()

def linear_val(train_loader, test_loader, test_sampler, epoch,args):

	### -----------> check if clip
	audio_model = models.CAVMAEFT_BASE(label_dim=args.n_class)
	
	### <--------
	mdl_weight = torch.load("%s/models/audio_model.%d.pth" % (args.exp_dir, epoch), map_location='cpu')
	print('#### MLP Val pretrained weights from ', "%s/models/audio_model.%d.pth" % (args.exp_dir, epoch))
	if not isinstance(audio_model, torch.nn.DataParallel):
		audio_model = torch.nn.DataParallel(audio_model)


	miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
	print(miss)
	audio_model = audio_model.module.to(torch.device('cpu'))

	device = torch.device("cuda", args.local_rank)
	audio_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(audio_model).to(device)
	audio_model = DDP(audio_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)    


	mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
			'mlp_head_mm.0.weight', 'mlp_head_mm.0.bias', 'mlp_head_mm.1.weight', 'mlp_head_mm.1.bias',
			'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
			'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
			'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight', 'mlp_head_concat.1.bias']
	mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
	base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
	mlp_params = [i[1] for i in mlp_params]
	base_params = [i[1] for i in base_params]

	# if freeze the pretrained parameters and only train the newly initialized model (linear probing)
	# if args.freeze_base == True:
	# print('Pretrained backbone parameters are frozen.')
	for param in base_params:
		param.requires_grad = False

	trainables = [p for p in audio_model.parameters() if p.requires_grad]

	print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
	print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

	# print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(args.head_lr))
	optimizer = torch.optim.Adam([{'params': base_params, 'lr': 5e-5}, {'params': mlp_params, 'lr': 5e-3}], weight_decay=5e-7, betas=(0.95, 0.999))
	base_lr = optimizer.param_groups[0]['lr']
	mlp_lr = optimizer.param_groups[1]['lr']

	scaler = GradScaler()
	epoch = 0
	loss_fn = nn.BCEWithLogitsLoss()
	while epoch < 5:
		# global_step=0
		for i, (a_input, v_input, labels) in enumerate(train_loader):

			if args.rank == 0:
				if i%10 ==0:
					print("MLP Train index: {}/{}".format(i,len(train_loader)))


			B = a_input.size(0)
			a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			with autocast():
				
				audio_output = audio_model(a_input, v_input, 'joint_av')
				loss = loss_fn(audio_output, labels)

			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

		epoch+=1
	#### ------> Val for few training <----------###
	print('start MLP validation')
	# stats, valid_loss = validate(audio_model, test_loader, args)
	stats, valid_loss = validate_mlp(audio_model, test_loader, test_sampler, 'joint_av' , args)
	mAP = np.mean([stat['AP'] for stat in stats])
	mAUC = np.mean([stat['auc'] for stat in stats])

	stats, valid_loss = validate_mlp(audio_model, test_loader, test_sampler, 'audioonly' , args)
	mAP_audio = np.mean([stat['AP'] for stat in stats])
	mAUC_audio = np.mean([stat['auc'] for stat in stats])


	stats, valid_loss = validate_mlp(audio_model, test_loader, test_sampler, 'videoonly' , args)
	mAP_video = np.mean([stat['AP'] for stat in stats])
	mAUC_video = np.mean([stat['auc'] for stat in stats])
	




	print("MLP mAP: {:.6f}".format(mAP))
	print("MLP AUC: {:.6f}".format(mAUC))

	print("MLP audio mAP: {:.6f}".format(mAP_audio))
	print("MLP audio AUC: {:.6f}".format(mAUC_audio))

	print("MLP video mAP: {:.6f}".format(mAP_video))
	print("MLP video AUC: {:.6f}".format(mAUC_video))
	if args.wandb:
		if args.rank == 0:
		
			wandb.log({"MLP-mAp-a+v": mAP})
			wandb.log({"MLP-mAUC-a+v": mAUC})

			wandb.log({"MLP-mAp-audio": mAP_audio})
			wandb.log({"MLP-mAUC-audio": mAUC_audio})

			wandb.log({"MLP-mAp-video": mAP_video})
			wandb.log({"MLP-mAUC-video": mAUC_video})


def validate(audio_model, val_loader, val_sampler, args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_time = AverageMeter()
	# if not isinstance(audio_model, nn.DataParallel):
	#     audio_model = nn.DataParallel(audio_model)
	# audio_model = audio_model.to(device)
	audio_model.eval()

	end = time.time()
	A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
	with torch.no_grad():
		for i, (a_input, v_input, _) in enumerate(val_loader):
			a_input = a_input.to(device)
			v_input = v_input.to(device)

			if args.rank == 0:
				if i%50 ==0:
					print("Val index: {}/{}".format(i,len(val_loader)))

			with autocast():
				loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
				loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
			A_loss.append(loss.to('cpu').detach())
			A_loss_mae.append(loss_mae.to('cpu').detach())
			A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
			A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
			A_loss_c.append(loss_c.to('cpu').detach())
			A_c_acc.append(c_acc.to('cpu').detach())


			batch_time.update(time.time() - end)
			end = time.time()




		loss = np.mean(A_loss)
		loss_mae = np.mean(A_loss_mae)
		loss_mae_a = np.mean(A_loss_mae_a)
		loss_mae_v = np.mean(A_loss_mae_v)
		loss_c = np.mean(A_loss_c)
		c_acc = np.mean(A_c_acc)

	return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc

def validate_mlp(audio_model, val_loader, val_sampler, mode, args, output_pred=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_time = AverageMeter()
	# if not isinstance(audio_model, nn.DataParallel):
	#     audio_model = nn.DataParallel(audio_model)
	# audio_model = audio_model.to(device)
	audio_model.eval()

	end = time.time()
	A_predictions, A_targets, A_loss = [], [], []
	yb_predictions = []
	yb_predictions_new = []
	yb_labels = []
	loss_fn = nn.BCEWithLogitsLoss()
	with torch.no_grad():
		for i, (a_input, v_input, labels) in enumerate(val_loader):
			a_input = a_input.to(device)
			v_input = v_input.to(device)

			if args.rank == 0:
				if i%50 ==0:
					print("Val index: {}/{}".format(i,len(val_loader)))

			with autocast():
				audio_output = audio_model(a_input, v_input, mode, is_eval=True)
				# audio_output, new_out = audio_model(a_input, v_input, args.ftmode, is_eval=True)
			

			yb_predictions.append(audio_output)
			# yb_predictions_new.append(new_out)

			yb_labels.append(labels.to(device))

  

			labels = labels.to(device)
			loss = loss_fn(audio_output.mean(dim=1), labels)
			A_loss.append(loss.to('cpu').detach())


		loss = np.mean(A_loss)

		audio_output = distributed_concat(torch.concat(yb_predictions, dim=0), len(val_sampler.dataset))
		# new_out = distributed_concat(torch.concat(yb_predictions_new, dim=0), len(val_sampler.dataset))


		target = distributed_concat(torch.concat(yb_labels, dim=0), len(val_sampler.dataset))

		# audio_output = torch.nn.functional.sigmoid(audio_output.float()) # !!!!!yb
		audio_output = torch.sigmoid(audio_output.float()) # !!!!!yb
		
		

		### -------->
		# new_out = torch.nn.functional.sigmoid(new_out.float()) # !!!!!yb
		# audio_output = (audio_output + new_out)/2
		##3 <--------

		if args.rank == 0:
			print("Final Val Shape: {}".format(target.shape))
		stats = calculate_stats(audio_output.mean(dim=1).to('cpu').detach(), target.to('cpu'))

	if output_pred == False:
		return stats, loss
	else:
		# used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
		return stats, audio_output, target