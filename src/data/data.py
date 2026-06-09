import torch
import numpy as np
import torch.utils.data
from typing import Dict
from data.data_provider import PEMSFLOWProvider, ENERGYProvider, BasicDataset
from data.scaler import StandardScaler
import os

data_dict = {
    'PEMS08FLOW': PEMSFLOWProvider,
    'PEMS04FLOW': PEMSFLOWProvider,
    'PEMS03FLOW': PEMSFLOWProvider,
    'PEMS07FLOW': PEMSFLOWProvider,
    'ENERGYFLOW': ENERGYProvider,
    # 'PEMS08MISSING': PEMSMISSINGProvider,
    # 'PEMS04MISSING': PEMSMISSINGProvider,
    # 'PEMS03MISSING': PEMSMISSINGProvider,
    # 'PEMS07MISSING': PEMSMISSINGProvider,
    # 'NYCTAXI': NYCTAXIProvider,
    # 'CHITAXI': NYCTAXIProvider,
}


def data_loader(dataset, batch_size, shuffle=True, drop_last=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def load_data(dataset, batch_size, sample_len, output_len, window_size, \
              input_dim, output_dim, \
              train_ratio, val_ratio, data_path, adj_path, target_strategy, few_shot=1, node_shuffle_seed=None):

    data_dir = os.path.dirname(os.path.abspath(data_path))
    if few_shot < 1:
        saved_filename = f"{dataset}_fewshot{few_shot}_datasets.pt"
    elif sample_len != 12 or output_len != 12:
        saved_filename = f"{dataset}_sl{sample_len}_pl{output_len}_datasets.pt" 
    else:
        saved_filename = f"{dataset}_datasets.pt"
    saved_path = os.path.join(data_dir, saved_filename)
    dataprovider = data_dict[dataset](data_path, adj_path, dataset, node_shuffle_seed)

    if os.path.exists(saved_path):
        print(f"[load_data] Loading saved data from: {saved_path}")
        saved_data = torch.load(saved_path)
        
        train_data = saved_data['train_dataset']
        train_set = BasicDataset(
            history=train_data['history'].cuda(),
            history_week_avg=train_data['history_week_avg'].cuda(),
            history_day_avg=train_data['history_day_avg'].cuda(),
            history_month_avg=train_data['history_month_avg'].cuda(),
            target=train_data['target'].cuda(),
            target_week_avg=train_data['target_week_avg'].cuda(),
            target_day_avg=train_data['target_day_avg'].cuda(),
            target_month_avg=train_data['target_month_avg'].cuda(),
            timestamp=train_data['timestamp'].cuda(),
            training=True
        )
        
        val_data = saved_data['val_dataset']
        val_set = BasicDataset(
            history=val_data['history'].cuda(),
            history_week_avg=val_data['history_week_avg'].cuda(),
            history_day_avg=val_data['history_day_avg'].cuda(),
            history_month_avg=val_data['history_month_avg'].cuda(),
            target=val_data['target'].cuda(),
            target_week_avg=val_data['target_week_avg'].cuda(),
            target_day_avg=val_data['target_day_avg'].cuda(),
            target_month_avg=val_data['target_month_avg'].cuda(),
            timestamp=val_data['timestamp'].cuda()
        )
        
        test_data = saved_data['test_dataset']
        test_set = BasicDataset(
            history=test_data['history'].cuda(),
            history_week_avg=test_data['history_week_avg'].cuda(),
            history_day_avg=test_data['history_day_avg'].cuda(),
            history_month_avg=test_data['history_month_avg'].cuda(),
            target=test_data['target'].cuda(),
            target_week_avg=test_data['target_week_avg'].cuda(),
            target_day_avg=test_data['target_day_avg'].cuda(),
            target_month_avg=test_data['target_month_avg'].cuda(),
            timestamp=test_data['timestamp'].cuda()
        )
        
        mean = [torch.tensor(m) for m in saved_data['scaler_mean']]
        std = [torch.tensor(s) for s in saved_data['scaler_std']]
        scaler = StandardScaler(mean, std)
        
        print(f"[load_data] Loaded from file successfully!")
        
    else:
        print(f"[load_data] Saved file not found, computing datasets (this may take a while)...")
        
        train_set, val_set, test_set = dataprovider.getdataset(
            sample_len=sample_len, output_len=output_len, window_size=window_size,
            input_dim=input_dim, output_dim=output_dim,
            train_ratio=train_ratio, val_ratio=val_ratio, few_shot=few_shot
        )
        
        scaler = dataprovider.scaler
        
        scaler_mean = [m.item() if hasattr(m, 'item') else m for m in scaler.mean]
        scaler_std = [s.item() if hasattr(s, 'item') else s for s in scaler.std]
        
        saved_data = {
            'train_dataset': {
                'history': train_set.history.cpu(),
                'history_week_avg': train_set.history_week_avg.cpu(),
                'history_day_avg': train_set.history_day_avg.cpu(),
                'history_month_avg': train_set.history_month_avg.cpu(),
                'target': train_set.target.cpu(),
                'target_week_avg': train_set.target_week_avg.cpu(),
                'target_day_avg': train_set.target_day_avg.cpu(),
                'target_month_avg': train_set.target_month_avg.cpu(),
                'timestamp': train_set.timestamp.cpu(),
            },
            'val_dataset': {
                'history': val_set.history.cpu(),
                'history_week_avg': val_set.history_week_avg.cpu(),
                'history_day_avg': val_set.history_day_avg.cpu(),
                'history_month_avg': val_set.history_month_avg.cpu(),
                'target': val_set.target.cpu(),
                'target_week_avg': val_set.target_week_avg.cpu(),
                'target_day_avg': val_set.target_day_avg.cpu(),
                'target_month_avg': val_set.target_month_avg.cpu(),
                'timestamp': val_set.timestamp.cpu(),
            },
            'test_dataset': {
                'history': test_set.history.cpu(),
                'history_week_avg': test_set.history_week_avg.cpu(),
                'history_day_avg': test_set.history_day_avg.cpu(),
                'history_month_avg': test_set.history_month_avg.cpu(),
                'target': test_set.target.cpu(),
                'target_week_avg': test_set.target_week_avg.cpu(),
                'target_day_avg': test_set.target_day_avg.cpu(),
                'target_month_avg': test_set.target_month_avg.cpu(),
                'timestamp': test_set.timestamp.cpu(),
            },
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std,
        }
        
        # Save to file
        print(f"[load_data] Saving to: {saved_path}")
        torch.save(saved_data, saved_path)
        file_size = os.path.getsize(saved_path) / (1024 * 1024)
        print(f"[load_data] Saved! Size: {file_size:.2f} MB")

    train_loader = data_loader(train_set, batch_size=batch_size)
    val_loader = data_loader(val_set, batch_size=batch_size)
    test_loader = data_loader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    node_num, features = dataprovider.node_num, dataprovider.features
    adj_mx, distance_mx = dataprovider.getadj()

    return train_loader, val_loader, test_loader, \
           scaler, node_num, features, \
           adj_mx, distance_mx
