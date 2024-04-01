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
from torch.nn.parallel import DistributedDataParallel as DDP

from ipdb import set_trace
import wandb
import torch.nn.functional as F
import random
from torch.profiler import profile, record_function, ProfilerActivity



def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def train(audio_model, train_loader, test_loader, test_sampler, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir



    device = torch.device("cuda", args.local_rank)
    audio_model = audio_model.to(device)


    mlp_params = []
    mm_layer_params = []
    base_params = []
    for name, param in audio_model.named_parameters():
        if 'mlp_head' in name:
            print('mlp_params: ', name)
            mlp_params.append([name,param])
        elif 'mm_layer' in name:
            print('mm_layer: ', name)
            mm_layer_params.append([name,param])
        else:
            print('others: ', name)
            base_params.append([name,param])
    

    # base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    mm_layer_params = [i[1] for i in mm_layer_params]

    # if freeze the pretrained parameters and only train the newly initialized model (linear probing)
    if args.freeze_base == True:
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(args.head_lr))
    optimizer = torch.optim.Adam([
                                    {'params': base_params, 'lr': args.lr}, 
                                    {'params': mlp_params, 'lr': args.lr * args.head_lr},
                                    {'params': mm_layer_params, 'lr': args.lr * args.mm_lr}
                                ]
                                , weight_decay=5e-7, betas=(0.95, 0.999))

    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)


    audio_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(audio_model).to(device)
    audio_model = DDP(audio_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)    


    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    # only for preliminary test, formal exps should use fixed learning rate scheduler
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])
    audio_model.train()
    count = 0
    loss_dis = nn.KLDivLoss(reduction="batchmean")
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (a_input, v_input, labels) in enumerate(train_loader):

            prob = random.uniform(0, 1) 

            B = a_input.size(0)
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            audio_output, out_a, out_v = audio_model(a_input, v_input, args.ftmode)


        
            with autocast():
                
                if args.ftmode == 'mm_grad':
                    audio_output, out_a, out_v = audio_model(a_input, v_input, args.ftmode)

                    if prob > 0.5:
                        loss = loss_fn(audio_output, labels) #+ args.dis_w*mydis_loss
                    elif prob < 0.25:
                        loss = loss_fn(out_a, labels)
                    else:
                        loss = loss_fn(out_v, labels)
                else:
                    audio_output = audio_model(a_input, v_input, args.ftmode)
                    loss = loss_fn(audio_output, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')

        stats, valid_loss = validate(audio_model, test_loader, test_sampler, args)


        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
        

        if args.wandb:
            if args.local_rank == 0:
                if main_metrics == 'acc':
                    # wandb.log({"val-audio": acc_a})
                    # wandb.log({"val-visual": acc_v})
                    wandb.log({"val-AV": acc})
                    # wandb.log({"val-AV_yb": acc_av_yb})
                else:
                    # wandb.log({"mAp-val-audio": mAP_a})
                     wandb.log({"mAp-val-audio": mAP})
                    # wandb.log({"mAp-visual": mAP})
                    # wandb.log({"mAp-AV_yb": mAP})


        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')


        count += 1
        if mAP > best_mAP:
            count = 0
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

                if args.wandb:
                    if args.local_rank == 0:
                        wandb.log({"val-best-av": best_mAP})

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

                if args.wandb:
                    if args.local_rank == 0:
                        wandb.log({"val-best-av": best_acc})

        ### early stop
        if count == 3:
            exit()

        if best_epoch == epoch:
            if args.local_rank == 0:
                torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
                torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

                if main_metrics == 'mAP':
                    torch.save(audio_model.state_dict(), "%s/models/audio_model.%.3f.pth" % (exp_dir, best_mAP))
                else:
                    torch.save(audio_model.state_dict(), "%s/models/audio_model.%.3f.pth" % (exp_dir, best_acc))
        if args.save_model == True:
            if args.local_rank == 0:
                torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def validate(audio_model, val_loader, val_sampler, args, output_pred=False):
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

    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            if args.local_rank == 0:
                if i%50 ==0:
                    print("Val index: {}/{}".format(i,len(val_loader)))

            with autocast():
                audio_output = audio_model(a_input, v_input, args.ftmode_test, is_eval=True)
                # audio_output, new_out = audio_model(a_input, v_input, args.ftmode, is_eval=True)
            

            yb_predictions.append(audio_output)
            # yb_predictions_new.append(new_out)

            yb_labels.append(labels.to(device))

  

            labels = labels.to(device)
            loss = args.loss_fn(audio_output.mean(dim=1), labels)
            A_loss.append(loss.to('cpu').detach())


        loss = np.mean(A_loss)

        audio_output = distributed_concat(torch.concat(yb_predictions, dim=0), len(val_sampler.dataset))
        # new_out = distributed_concat(torch.concat(yb_predictions_new, dim=0), len(val_sampler.dataset))


        target = distributed_concat(torch.concat(yb_labels, dim=0), len(val_sampler.dataset))

        audio_output = torch.nn.functional.sigmoid(audio_output.float()) # !!!!!yb
        


        if args.local_rank == 0:
            print("Final Val Shape: {}".format(target.shape))
        stats = calculate_stats(audio_output.mean(dim=1).to('cpu').detach(), target.to('cpu'))

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, audio_output, target