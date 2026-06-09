import torch
import numpy as np
import torch.utils.data
from utils.utils import get_adjacency_matrix_2direction, get_adjacency_matrix
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from data.scaler import StandardScaler, MinMaxScaler
import os

def generate_sample_by_sliding_window(data, sample_len, step=1):
    sample = []
    for i in range(0, data.shape[0] - sample_len, step):
        sample.append(torch.unsqueeze(data[i:i+sample_len], 0))
    
    if (data.shape[0] - sample_len) % step != 0:
        sample.append(torch.unsqueeze(data[-sample_len:], 0))
    
    sample = torch.concat(sample, dim=0)
    return sample


class BasicDataset(torch.utils.data.Dataset):

    history: torch.Tensor              # (B, sample_len, node_num, features)
    history_week_avg: torch.Tensor     # (B, sample_len, node_num, features) - weekly average
    history_day_avg: torch.Tensor      # (B, sample_len, node_num, features) - daily average
    history_month_avg: torch.Tensor    # (B, sample_len, node_num, features) - monthly average
    target: torch.Tensor               # (B, output_len, node_num, output_dim)
    target_week_avg: torch.Tensor      # (B, output_len, node_num, output_dim) - weekly average target
    target_day_avg: torch.Tensor       # (B, output_len, node_num, output_dim) - daily average target
    target_month_avg: torch.Tensor     # (B, output_len, node_num, output_dim) - monthly average target
    timestamp: torch.Tensor            # (B, window_size, 5)

    def __init__(self, history, history_week_avg, history_day_avg, history_month_avg,
                 target, target_week_avg, target_day_avg, target_month_avg, 
                 timestamp, training=False) -> None:
        self.history = history
        self.history_week_avg = history_week_avg
        self.history_day_avg = history_day_avg
        self.history_month_avg = history_month_avg
        self.target = target
        self.target_week_avg = target_week_avg
        self.target_day_avg = target_day_avg
        self.target_month_avg = target_month_avg
        self.timestamp = timestamp
        self.training = training

    def __len__(self):
        return self.history.shape[0]
    
    def __getitem__(self, index):
        return (self.history[index], 
                self.history_week_avg[index], self.history_day_avg[index], self.history_month_avg[index],
                self.target[index], 
                self.target_week_avg[index], self.target_day_avg[index], self.target_month_avg[index],
                self.timestamp[index])


class DataProvider():
    node_num: int
    features: int
    data: torch.Tensor       # (T, node_num, features) - dữ liệu gốc
    timestamp: torch.Tensor  # (T, 5) - [month, day, weekday, hour, minute]

    def __init__(self, data_path, adj_path, dataset, node_shuffle_seed=None) -> None:
        self.dataset = dataset

        self.data, self.node_num, self.features, \
        self.adj_mx, self.distance_mx, \
        self.timestamp = self.read_data(data_path, adj_path)

        if node_shuffle_seed is not None:
            rdm = np.random.RandomState(node_shuffle_seed)
            idx = np.arange(self.node_num)
            rdm.shuffle(idx)
            idx = torch.from_numpy(idx)
            self.data = self.data[:, idx, :]
            self.adj_mx = self.adj_mx[idx, :][:, idx]

    def compute_weekly_average(self, train_data, train_timestamp):
        T_full = self.data.shape[0]
        
        # Tính weekdaytime cho tập train
        weekday = train_timestamp[:, 2]      # 0-6
        hour = train_timestamp[:, 3]         # 0-23
        minute = train_timestamp[:, 4]       # 0-59
        train_weekdaytime = weekday * 288 + (hour * 60 + minute) // 5

        weekly_avg_dict = {}
        
        # Tính trung bình cho từng node, từng feature
        for node_idx in range(self.node_num):
            for feature_idx in range(self.features):

                flow_values = train_data[:, node_idx, feature_idx].cpu().numpy()
                wdt_values = train_weekdaytime.cpu().numpy()

                df = pd.DataFrame({
                    'weekdaytime': wdt_values,
                    'flow': flow_values
                })
                
                def get_mean_without_null(data):
                    return data[data != 0].mean() if (data != 0).any() else 0
                
                avg_by_wdt = df.groupby('weekdaytime')['flow'].apply(get_mean_without_null)
                
                for wdt, avg_val in avg_by_wdt.items():
                    weekly_avg_dict[(node_idx, feature_idx, wdt)] = avg_val
        
        weekly_avg_full = torch.zeros_like(self.data)
        
        full_weekday = self.timestamp[:, 2]
        full_hour = self.timestamp[:, 3]
        full_minute = self.timestamp[:, 4]
        full_wdt = (full_weekday * 288 + (full_hour * 60 + full_minute) // 5).cpu().numpy()
        
        for t in range(T_full):
            wdt = full_wdt[t]
            for node_idx in range(self.node_num):
                for feature_idx in range(self.features):
                    key = (node_idx, feature_idx, wdt)
                    weekly_avg_full[t, node_idx, feature_idx] = weekly_avg_dict.get(key, 0)
        
        return weekly_avg_full

    def compute_daily_average(self, train_data, train_timestamp):
        T_full = self.data.shape[0]
        hour = train_timestamp[:, 3]         # 0-23
        minute = train_timestamp[:, 4]       # 0-59
        train_daytime = (hour * 60 + minute) // 5  # 288 slots

        daily_avg_dict = {}
        
        for node_idx in range(self.node_num):
            for feature_idx in range(self.features):
                flow_values = train_data[:, node_idx, feature_idx].cpu().numpy()
                dt_values = train_daytime.cpu().numpy()

                df = pd.DataFrame({
                    'daytime': dt_values,
                    'flow': flow_values
                })
                
                def get_mean_without_null(data):
                    return data[data != 0].mean() if (data != 0).any() else 0
                
                avg_by_dt = df.groupby('daytime')['flow'].apply(get_mean_without_null)
                
                for dt, avg_val in avg_by_dt.items():
                    daily_avg_dict[(node_idx, feature_idx, dt)] = avg_val
        
        daily_avg_full = torch.zeros_like(self.data)
        
        full_hour = self.timestamp[:, 3]
        full_minute = self.timestamp[:, 4]
        full_dt = ((full_hour * 60 + full_minute) // 5).cpu().numpy()
        
        for t in range(T_full):
            dt = full_dt[t]
            for node_idx in range(self.node_num):
                for feature_idx in range(self.features):
                    key = (node_idx, feature_idx, dt)
                    daily_avg_full[t, node_idx, feature_idx] = daily_avg_dict.get(key, 0)
        
        return daily_avg_full

    def compute_monthly_average(self, train_data, train_timestamp):
        T_full = self.data.shape[0]
        
        day = train_timestamp[:, 1]          # 1-31
        hour = train_timestamp[:, 3]         # 0-23
        minute = train_timestamp[:, 4]       # 0-59
        train_monthdaytime = day * 288 + (hour * 60 + minute) // 5

        monthly_avg_dict = {}
        
        for node_idx in range(self.node_num):
            for feature_idx in range(self.features):
                flow_values = train_data[:, node_idx, feature_idx].cpu().numpy()
                mdt_values = train_monthdaytime.cpu().numpy()

                df = pd.DataFrame({
                    'monthdaytime': mdt_values,
                    'flow': flow_values
                })
                
                def get_mean_without_null(data):
                    return data[data != 0].mean() if (data != 0).any() else 0
                
                avg_by_mdt = df.groupby('monthdaytime')['flow'].apply(get_mean_without_null)
                
                for mdt, avg_val in avg_by_mdt.items():
                    monthly_avg_dict[(node_idx, feature_idx, mdt)] = avg_val
        
        monthly_avg_full = torch.zeros_like(self.data)
        
        full_day = self.timestamp[:, 1]
        full_hour = self.timestamp[:, 3]
        full_minute = self.timestamp[:, 4]
        full_mdt = (full_day * 288 + (full_hour * 60 + full_minute) // 5).cpu().numpy()
        
        for t in range(T_full):
            mdt = full_mdt[t]
            for node_idx in range(self.node_num):
                for feature_idx in range(self.features):
                    key = (node_idx, feature_idx, mdt)
                    monthly_avg_full[t, node_idx, feature_idx] = monthly_avg_dict.get(key, 0)
        
        return monthly_avg_full

    def getdataset(self, sample_len, output_len, window_size,
                   input_dim, output_dim,
                   train_ratio, val_ratio, few_shot=1):
        
        self.data = self.data.float().cuda()
        self.timestamp = self.timestamp.cuda()

        all_len = self.data.shape[0]
        if few_shot != 1:
            train_len = int(all_len * few_shot)
        else:
            train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)

        train_range = [0, train_len]
        val_range = [train_len, train_len + val_len]
        test_range = [train_len + val_len, all_len]

        train_data = self.data[train_range[0]:train_range[1]]
        train_te = self.timestamp[train_range[0]:train_range[1]]
        
        # Compute all 3 types of averages
        weekly_avg = self.compute_weekly_average(train_data, train_te).cuda()
        daily_avg = self.compute_daily_average(train_data, train_te).cuda()
        monthly_avg = self.compute_monthly_average(train_data, train_te).cuda()

        scaler_data = self.data[train_range[0]:train_range[1]]
        dim = scaler_data.shape[-1]
        mean = [scaler_data[..., i:i+1].mean() for i in range(dim)]
        std = [scaler_data[..., i:i+1].std() for i in range(dim)]
        self.scaler = self.getscalerclass()(mean, std)

        # Training set
        train_data_normal = self.data[train_range[0]:train_range[1]]
        train_week_avg = weekly_avg[train_range[0]:train_range[1]]
        train_day_avg = daily_avg[train_range[0]:train_range[1]]
        train_month_avg = monthly_avg[train_range[0]:train_range[1]]
        
        train_sample = generate_sample_by_sliding_window(train_data_normal, sample_len=window_size)
        train_sample_week = generate_sample_by_sliding_window(train_week_avg, sample_len=window_size)
        train_sample_day = generate_sample_by_sliding_window(train_day_avg, sample_len=window_size)
        train_sample_month = generate_sample_by_sliding_window(train_month_avg, sample_len=window_size)
        
        train_x = train_sample[:, :sample_len, ..., :input_dim]
        train_x_week = train_sample_week[:, :sample_len, ..., :input_dim]
        train_x_day = train_sample_day[:, :sample_len, ..., :input_dim]
        train_x_month = train_sample_month[:, :sample_len, ..., :input_dim]
        train_y = train_sample[:, -output_len:, ..., :output_dim]
        train_y_week = train_sample_week[:, -output_len:, ..., :output_dim]
        train_y_day = train_sample_day[:, -output_len:, ..., :output_dim]
        train_y_month = train_sample_month[:, -output_len:, ..., :output_dim]
        
        train_x = self.scaler.transform(train_x)
        train_x_week = self.scaler.transform(train_x_week)
        train_x_day = self.scaler.transform(train_x_day)
        train_x_month = self.scaler.transform(train_x_month)

        train_y = self.scaler.transform(train_y)
        train_y_week = self.scaler.transform(train_y_week)
        train_y_day = self.scaler.transform(train_y_day)
        train_y_month = self.scaler.transform(train_y_month)
        
        train_te = generate_sample_by_sliding_window(train_te, sample_len=window_size)
        train_dataset = BasicDataset(
            history=train_x, 
            history_week_avg=train_x_week, history_day_avg=train_x_day, history_month_avg=train_x_month,
            target=train_y, 
            target_week_avg=train_y_week, target_day_avg=train_y_day, target_month_avg=train_y_month,
            timestamp=train_te, training=True
        )

        # Validation set
        val_data_normal = self.data[val_range[0]:val_range[1]]
        val_week_avg = weekly_avg[val_range[0]:val_range[1]]
        val_day_avg = daily_avg[val_range[0]:val_range[1]]
        val_month_avg = monthly_avg[val_range[0]:val_range[1]]
        val_te = self.timestamp[val_range[0]:val_range[1]]
        
        val_sample = generate_sample_by_sliding_window(val_data_normal, sample_len=window_size)
        val_sample_week = generate_sample_by_sliding_window(val_week_avg, sample_len=window_size)
        val_sample_day = generate_sample_by_sliding_window(val_day_avg, sample_len=window_size)
        val_sample_month = generate_sample_by_sliding_window(val_month_avg, sample_len=window_size)
        
        val_x = val_sample[:, :sample_len, ..., :input_dim]
        val_x_week = val_sample_week[:, :sample_len, ..., :input_dim]
        val_x_day = val_sample_day[:, :sample_len, ..., :input_dim]
        val_x_month = val_sample_month[:, :sample_len, ..., :input_dim]
        val_y = val_sample[:, -output_len:, ..., :output_dim]
        val_y_week = val_sample_week[:, -output_len:, ..., :output_dim]
        val_y_day = val_sample_day[:, -output_len:, ..., :output_dim]
        val_y_month = val_sample_month[:, -output_len:, ..., :output_dim]
        
        val_x = self.scaler.transform(val_x)
        val_x_week = self.scaler.transform(val_x_week)
        val_x_day = self.scaler.transform(val_x_day)
        val_x_month = self.scaler.transform(val_x_month)

        val_y = self.scaler.transform(val_y)
        val_y_week = self.scaler.transform(val_y_week)
        val_y_day = self.scaler.transform(val_y_day)
        val_y_month = self.scaler.transform(val_y_month)
        
        val_te = generate_sample_by_sliding_window(val_te, sample_len=window_size)
        val_dataset = BasicDataset(
            history=val_x, 
            history_week_avg=val_x_week, history_day_avg=val_x_day, history_month_avg=val_x_month,
            target=val_y, 
            target_week_avg=val_y_week, target_day_avg=val_y_day, target_month_avg=val_y_month,
            timestamp=val_te
        )

        # Test set
        test_data_normal = self.data[test_range[0]:test_range[1]]
        test_week_avg = weekly_avg[test_range[0]:test_range[1]]
        test_day_avg = daily_avg[test_range[0]:test_range[1]]
        test_month_avg = monthly_avg[test_range[0]:test_range[1]]
        test_te = self.timestamp[test_range[0]:test_range[1]]
        
        test_sample = generate_sample_by_sliding_window(test_data_normal, sample_len=window_size)
        test_sample_week = generate_sample_by_sliding_window(test_week_avg, sample_len=window_size)
        test_sample_day = generate_sample_by_sliding_window(test_day_avg, sample_len=window_size)
        test_sample_month = generate_sample_by_sliding_window(test_month_avg, sample_len=window_size)
        
        test_x = test_sample[:, :sample_len, ..., :input_dim]
        test_x_week = test_sample_week[:, :sample_len, ..., :input_dim]
        test_x_day = test_sample_day[:, :sample_len, ..., :input_dim]
        test_x_month = test_sample_month[:, :sample_len, ..., :input_dim]
        test_y = test_sample[:, -output_len:, ..., :output_dim]
        test_y_week = test_sample_week[:, -output_len:, ..., :output_dim]
        test_y_day = test_sample_day[:, -output_len:, ..., :output_dim]
        test_y_month = test_sample_month[:, -output_len:, ..., :output_dim]
        
        test_x = self.scaler.transform(test_x)
        test_x_week = self.scaler.transform(test_x_week)
        test_x_day = self.scaler.transform(test_x_day)
        test_x_month = self.scaler.transform(test_x_month)

        test_y = self.scaler.transform(test_y)
        test_y_week = self.scaler.transform(test_y_week)
        test_y_day = self.scaler.transform(test_y_day)
        test_y_month = self.scaler.transform(test_y_month)
        
        test_te = generate_sample_by_sliding_window(test_te, sample_len=window_size)
        test_dataset = BasicDataset(
            history=test_x, 
            history_week_avg=test_x_week, history_day_avg=test_x_day, history_month_avg=test_x_month,
            target=test_y, 
            target_week_avg=test_y_week, target_day_avg=test_y_day, target_month_avg=test_y_month,
            timestamp=test_te
        )

        return train_dataset, val_dataset, test_dataset

    def getadj(self):
        return self.adj_mx, self.distance_mx
    
    def getscalerclass(self):
        return StandardScaler


def generatetimestamp(start, periods, freq):
    time = pd.date_range(start=start, periods=periods, freq=freq)
    
    month = np.reshape(time.month, (-1, 1))
    dayofmonth = np.reshape(time.day, (-1, 1))
    dayofweek = np.reshape(time.weekday, (-1, 1))
    hour = np.reshape(time.hour, (-1, 1))
    minute = np.reshape(time.minute, (-1, 1))
    
    timestamp = np.concatenate((month, dayofmonth, dayofweek, hour, minute), -1)
    timestamp = torch.tensor(timestamp)
    
    return timestamp


timestampfun = {
    'PEMS08': lambda T: generatetimestamp(start='20160701 00:00:00', periods=T, freq='5min'),
    'PEMS07': lambda T: generatetimestamp(start='20170501 00:00:00', periods=T, freq='5min'),
    'PEMS04': lambda T: generatetimestamp(start='20180101 00:00:00', periods=T, freq='5min'),
    'PEMS03': lambda T: generatetimestamp(start='20180901 00:00:00', periods=T, freq='5min'),
    'NYCTAXI': lambda T: generatetimestamp(start='20160401 00:00:00', periods=T, freq='30min'),
    'CHIBIKE': lambda T: generatetimestamp(start='20160401 00:00:00', periods=T, freq='30min'),
    'ENERGY': lambda T: generatetimestamp(start='20190101 00:00:00', periods=T, freq='1h'),
}


class PEMSFLOWProvider(DataProvider):

    def read_data(self, data_path, adj_path=None) -> None:
        data = torch.from_numpy(np.load(data_path)['data'][..., :])
        
        T, node_num, features = data.shape
        
        if 'PEMS03' in self.dataset:
            id_filename = adj_path.replace('csv', 'txt')
        else:
            id_filename = None
        
        adj_mx, distance_mx = get_adjacency_matrix(adj_path, node_num, id_filename)
        adj_mx = np.where(np.eye(node_num).astype('bool'), 1, adj_mx)
        
        timestamp = timestampfun[self.dataset[:6]](T)
        
        return data, node_num, features, adj_mx, distance_mx, timestamp


class NYCTAXIProvider(DataProvider):

    def read_data(self, data_path, adj_path=None) -> None:
        data = torch.from_numpy(np.load(data_path)['data'][..., :])
        data = np.transpose(data, (1, 0, 2))
        
        T, node_num, features = data.shape
        
        adj_mx = np.ones((node_num, node_num)).astype(np.float32)
        distance_mx = np.ones((node_num, node_num)).astype(np.float32)
        timestamp = timestampfun[self.dataset](T)
        
        return data, node_num, features, adj_mx, distance_mx, timestamp


class ENERGYProvider(DataProvider):

    def read_data(self, data_path, adj_path=None) -> None:
        # Load hourly energy wind data matrix
        raw_data = np.load(data_path)['x']
        if len(raw_data.shape) == 2:
            raw_data = np.expand_dims(raw_data, axis=-1)  # (T, node_num, 1)
        data = torch.from_numpy(raw_data)
        
        T, node_num, features = data.shape
        
        # Load adjacency matrix
        adj_mx = np.load(adj_path)['x']
        adj_mx = np.where(np.eye(node_num).astype('bool'), 1.0, adj_mx)
        # adj_mx = adj_mx.float()
        
        distance_mx = np.ones((node_num, node_num)).astype(np.float32)
        
        # Generate timestamps using the hourly configuration
        timestamp = timestampfun[self.dataset[:6]](T)
        
        return data, node_num, features, adj_mx, distance_mx, timestamp