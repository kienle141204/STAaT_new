import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import time
import json
from src.utils.utils import get_time_str,check_dir,draw_loss_line,draw_mape_node,get_randmask,get_block_mask, cal_shortest_path_length
from src.logger import getlogger
from src.model.model import AICLLM
from src.model.llm import GPT2, LLaMA7B, Transformer
from src.data.data import load_data
from src.utils.metrics import MAE_torch,RMSE_torch,MAPE_torch,MAPE_torch_node,cal_metrics
from src.utils.argsinit import InitArgs
from src.utils.csv_logger import log_result_to_csv
import copy
from torch.optim.lr_scheduler import ExponentialLR
import nni
import random
import string
import wandb
from datetime import datetime
from src.prompts import PROMPTS
wandb.login(key = 'c18f56f87b92b4296251b454a8556397e6153841')


random_str = lambda : ''.join(random.sample(string.ascii_letters + string.digits, 6))
seed=random.randint(0, 100000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def TrainEpoch(loader, model, optim, loss_fn, prompt_prefix, scaler, need_step: bool):
    if need_step:
        model.train()
    else:
        model.eval()

    loss_item = 0
    count = 0
    epoch_start = time.time()

    for input, input_week, input_day, target, target_week, target_day, timestamp in loader:
        # (B,T,N,F)
        B, T, N, F = input.shape
        input = input.permute(0,2,1,3).contiguous().view(B,N,-1)
        input_day = input_day.permute(0,2,1,3).contiguous().view(B,N,-1)
        input_week = input_week.permute(0,2,1,3).contiguous().view(B,N,-1)
        target_week = target_week.permute(0,2,1,3).contiguous().view(B,N,-1)
        target_day = target_day.permute(0,2,1,3).contiguous().view(B,N,-1)

        predict, other_loss = model(input, input_day, input_week, target_week, target_day, timestamp, prompt_prefix)

        predict = predict.view(B, N, -1, args.output_dim).permute(0, 2, 1, 3).contiguous()  #(B, T, N, F)
        predict = scaler.inverse_transform(predict)
        target = scaler.inverse_transform(target)

        loss = loss_fn(predict, target)

        loss_item += loss.item()
        count += 1

        if need_step:
            optim.zero_grad()

            L = loss

            for l in other_loss:
                L += l

            L.backward()

            optim.step()

    if count:
        loss_item /= count

    return loss_item, time.time() - epoch_start


def TestEpoch(loader, model, prompt_prefix, scaler, save=False):
    test_start = time.time()
    with torch.no_grad():
        model.eval()
        targets = []
        predicts = []

        for input, input_week, input_day, target, target_week, target_day, timestamp in loader:
            B, T, N, F = input.shape

            input = input.permute(0,2,1,3).contiguous().view(B,N,-1)
            input_day = input_day.permute(0,2,1,3).contiguous().view(B,N,-1)
            input_week = input_week.permute(0,2,1,3).contiguous().view(B,N,-1)
            target_week = target_week.permute(0,2,1,3).contiguous().view(B,N,-1)
            target_day = target_day.permute(0,2,1,3).contiguous().view(B,N,-1)

            predict, _ = model(input, input_day, input_week, target_week, target_day, timestamp, prompt_prefix)

            predict = predict.view(B, N, -1, args.output_dim).permute(0, 2, 1, 3).contiguous()

            targets.append(target.detach())
            predicts.append(predict.detach())

        targets = torch.concat(targets, dim=0)
        predicts = torch.concat(predicts, dim=0)

        predicts = scaler.inverse_transform(predicts)
        targets = scaler.inverse_transform(targets)

        mae_pred = MAE_torch(pred=predicts[:,-args.predict_len:],true=targets[:,-args.predict_len:])
        rmse_pred = RMSE_torch(pred=predicts[:,-args.predict_len:],true=targets[:,-args.predict_len:])
        mape_pred = MAPE_torch(pred=predicts[:,-args.predict_len:],true=targets[:,-args.predict_len:])

    if save:
        np.savez(os.path.join(LOG_DIR, 'test.npz'), targets=targets.cpu().numpy(), predicts=predicts.cpu().numpy())

    return mae_pred, rmse_pred, mape_pred, time.time() - test_start


def compute_macs(model, node_num, args, output_len):
    """Estimate MACs for one forward pass with batch_size=1. Returns -1 if thop is unavailable."""
    try:
        from thop import profile
        B, N = 1, node_num
        x_dummy       = torch.zeros(B, N, args.sample_len * args.input_dim).cuda()
        day_dummy     = torch.zeros(B, N, args.sample_len * args.input_dim).cuda()
        week_dummy    = torch.zeros(B, N, args.sample_len * args.input_dim).cuda()
        ya_week_dummy = torch.zeros(B, N, output_len * args.output_dim).cuda()
        ya_day_dummy  = torch.zeros(B, N, output_len * args.output_dim).cuda()
        ts_dummy      = torch.zeros(B, args.sample_len + output_len, 5).cuda()
        macs, _ = profile(
            model,
            inputs=(x_dummy, day_dummy, week_dummy, ya_week_dummy, ya_day_dummy, ts_dummy, None),
            verbose=False
        )
        return int(macs)
    except Exception as e:
        print(f"[MACs] Could not compute: {e}")
        return -1


def Train(args, mylogger, model, prompt_prefix, scaler):

    patience_count = 0

    max_epoch = args.epoch

    if args.zero_shot:
        max_epoch = 0

    lr = args.lr
    val_epoch = args.val_epoch
    test_epoch = args.test_epoch

    optim = torch.optim.AdamW([
        {'params': (p for name, p in model.named_parameters() if ('bias' not in name) and p.requires_grad), 'weight_decay': args.weight_decay},
        {'params': (p for name, p in model.named_parameters() if ('bias' in name) and p.requires_grad)}
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10, min_lr=1e-6)

    loss_fn = torch.nn.L1Loss()

    best_loss = 1e9
    best_model = copy.deepcopy(model.grad_state_dict())

    train_loss_line = {'x': [], 'y': []}
    val_loss_line = {'x': [], 'y': []}

    train_times = []
    test_times  = []

    for epoch in range(max_epoch):

        train_loss, train_time = TrainEpoch(train_loader, model, optim, loss_fn, prompt_prefix, scaler, need_step=True)
        train_times.append(train_time)

        train_loss_line['x'].append(epoch)
        train_loss_line['y'].append(train_loss)

        mylogger.info(f"epoch {epoch} train_loss:{train_loss:.4f}  train_time:{train_time:.2f}s")

        if epoch % val_epoch == 0:

            val_loss, _ = TrainEpoch(val_loader, model, optim, loss_fn, prompt_prefix, scaler, need_step=False)
            val_loss_line['x'].append(epoch)
            val_loss_line['y'].append(val_loss)
            wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss}, step=epoch)

            if val_loss < best_loss:
                patience_count = 0
                best_loss = val_loss
                best_model = copy.deepcopy(model.grad_state_dict())
            else:
                patience_count += 1

            if args.nni:
                nni.report_intermediate_result(val_loss)
            mylogger.info(f"[Validation] epoch {epoch} val_loss:{val_loss}")
            scheduler.step(val_loss)

        if epoch % test_epoch == 0:

            mae_pred, rmse_pred, mape_pred, test_time = TestEpoch(test_loader, model, prompt_prefix, scaler=scaler)
            test_times.append(test_time)

            if args.task in ['all', 'prediction']:
                mylogger.info(f"[Test][prediction] epoch {epoch} mae:{mae_pred} rmse:{rmse_pred} mape:{mape_pred}  test_time:{test_time:.2f}s")

        mylogger.info(f"[Scheduler] epoch {epoch} lr:{optim.param_groups[0]['lr']}")

        if patience_count >= args.patience:
            mylogger.info('early stop')
            break


    if args.nni:
        nni.report_final_result(best_loss)

    model.load_state_dict(best_model, strict=False)

    mae_pred, rmse_pred, mape_pred, final_test_time = TestEpoch(test_loader, model, prompt_prefix, scaler, save=args.save_result)
    test_times.append(final_test_time)
    wandb.log({"Best Test Prediction MAE": mae_pred, "Best Test Prediction RMSE": rmse_pred, "Best Test Prediction MAPE": mape_pred}, step=0)

    if args.task in ['all', 'prediction']:
        mylogger.info(f"[Test][prediction] best model mae:{mae_pred} rmse:{rmse_pred} mape:{mape_pred}")

    draw_loss_line(train_loss_line, val_loss_line, os.path.join(LOG_DIR, 'loss.png'))

    avg_train_time = sum(train_times) / len(train_times) if train_times else 0.0
    avg_test_time  = sum(test_times)  / len(test_times)  if test_times  else 0.0

    return mae_pred, rmse_pred, mape_pred, avg_train_time, avg_test_time


def getllm(args):
    if args.model == 'gpt2':
        basemodel = GPT2(args.lora, args.ln_grad, args.llm_layers)
    elif args.model == 'llama7b':
        basemodel = LLaMA7B(args.lora, args.ln_grad, args.llm_layers)
    elif args.model == 'transformer':
        basemodel = Transformer(args.causal, args.lora, args.ln_grad, args.llm_layers)
    else:
        raise ValueError(f"Model '{args.model}' is not supported. Please use --model 'gpt2' or 'llama7b'.")

    return basemodel

if __name__ == '__main__':

    args = InitArgs()
    wandb.init(project=f"AIC-LLM_{args.dataset}", name=f"{args.desc}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}", config=vars(args))

    output_len = args.predict_len
    window_size = args.sample_len + args.predict_len
    if args.task == 'all':
        output_len += args.sample_len
    elif args.task == 'imputation':
        output_len = args.sample_len
        window_size -= args.predict_len

    if args.nni:
        params = nni.get_next_parameter()
        args.time_token_dim = params['time_token_dim']
        args.node_emb_dim = params['node_emb_dim']
        args.trunc_k = params['trunc_k']

    basemodel = getllm(args)

    train_loader, val_loader, test_loader,\
           scaler,  node_num, features , \
           adj_mx, distance_mx = load_data(dataset=args.dataset, batch_size=args.batch_size, sample_len= args.sample_len, output_len = output_len, window_size = window_size,\
                                           input_dim = args.input_dim, output_dim = args.output_dim,\
                                           train_ratio = args.train_ratio, val_ratio = args.val_ratio, \
                                            data_path = args.data_path , adj_path = args.adj_filename, \
                                            target_strategy = args.target_strategy, \
                                           few_shot = args.few_shot, node_shuffle_seed = args.node_shuffle_seed)
    #distance_mx = cal_shortest_path_length(adj_mx, distance_mx)

    steps_per_day = 144 if args.dataset[:6] == 'ENERGY' else 288

    # Get prompt template (not tokenized yet)
    prompt_template = None
    if args.dataset[:6] in PROMPTS:
        print(f"Loading prompt template for {args.dataset}...")
        prompt_template = PROMPTS[args.dataset[:6]]
        print("Prompt template loaded (will be embedded once)")

    if args.prompt_prefix is not None:
        prompt_template = args.prompt_prefix


    LOG_DIR = os.path.join(args.log_root,f'{get_time_str()}_{args.desc}_{random_str()}')

    check_dir(LOG_DIR,mkdir=True)

    logpath = os.path.join(LOG_DIR,f'experiments.log')
    modelpath = os.path.join(LOG_DIR,f'{get_time_str()}_{args.desc}.pth')

    mylogger = getlogger(logpath)

    mylogger.info(args)

    model = AICLLM(basemodel=basemodel, sample_len= args.sample_len, output_len = output_len, \
                    input_dim = args.input_dim , output_dim = args.output_dim , \
                     node_emb_dim=args.node_emb_dim , \
                    sag_dim = args.sag_dim, sag_tokens = args.sag_tokens, \
                     adj_mx = adj_mx, dis_mx = distance_mx, \
                    use_node_embedding = args.node_embedding ,use_time_token= args.time_token, \
                    prompt_template = prompt_template, \
                    task_type = args.task, use_instruction = args.use_instruction, \
                    use_anchor_day=args.use_anchor_day, use_anchor_week=args.use_anchor_week, \
                    anchor_week_loss_weight=args.anchor_week_loss_weight, anchor_day_loss_weight=args.anchor_day_loss_weight, \
                    anchor_loss_type=args.anchor_loss_type, \
                    use_sandglassAttn = args.sandglassAttn, dropout = args.dropout, trunc_k = args.trunc_k, t_dim = args.t_dim,wo_conloss=args.wo_conloss, wo_conloss1=args.wo_conloss1, wo_conloss2=args.wo_conloss2,
                    ablation_drop_token=args.ablation_drop_token,
                    steps_per_day=steps_per_day).cuda()


    if not args.from_pretrained_model is None:
        model.load(args.from_pretrained_model)

    if args.zero_shot and args.from_pretrained_model is None :
        mylogger.info(f'Please specify pretrained model when test zero-shot')
        exit()

    mylogger.info(model)
    total_params, total_trainable_params = model.params_num()
    mylogger.info(f'total_params:{total_params}    total_trainable_params:{total_trainable_params}')
    mylogger.info(model.grad_state_dict().keys())

    # Compute MACs (requires `pip install thop`)
    macs = compute_macs(model, node_num, args, output_len)
    mylogger.info(f'MACs: {macs:,}' if macs >= 0 else 'MACs: unavailable (install thop)')
    wandb.log({"total_params": total_params, "total_trainable_params": total_trainable_params, "macs": macs})

    mae, rmse, mape, avg_train_time, avg_test_time = Train(args, mylogger, model, None, scaler)

    mylogger.info(f'avg_train_time_per_epoch:{avg_train_time:.2f}s    avg_test_time_per_epoch:{avg_test_time:.2f}s')
    wandb.log({"avg_train_time_s": avg_train_time, "avg_test_time_s": avg_test_time})

    # Log to CSV
    csv_path = log_result_to_csv(args, mae, rmse, mape, total_params, total_trainable_params,
                                 macs=macs, avg_train_time=avg_train_time, avg_test_time=avg_test_time)
    mylogger.info(f"Results saved to CSV: {csv_path}")

    # Save JSON summary for easy programmatic access
    summary = {
        "run_id": os.path.basename(LOG_DIR),
        "dataset": args.dataset,
        "model": args.model,
        "params": {
            "total_params": total_params,
            "total_trainable_params": total_trainable_params,
            "macs": macs,
        },
        "timing": {
            "avg_train_time_per_epoch_s": round(avg_train_time, 3),
            "avg_test_time_per_epoch_s":  round(avg_test_time,  3),
        },
        "metrics": {
            "mae":  float(mae)  if mae  is not None else None,
            "rmse": float(rmse) if rmse is not None else None,
            "mape": float(mape) if mape is not None else None,
        },
        "args": vars(args),
    }
    summary_path = os.path.join(LOG_DIR, 'results_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    mylogger.info(f"JSON summary saved: {summary_path}")

    model.save(modelpath)
