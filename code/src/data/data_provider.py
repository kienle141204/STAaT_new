import torch
import numpy as np
import torch.utils.data
from src.utils.utils import get_adjacency_matrix_2direction, get_adjacency_matrix
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from src.data.scaler import StandardScaler, MinMaxScaler
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
    target: torch.Tensor               # (B, output_len, node_num, output_dim)
    target_week_avg: torch.Tensor      # (B, output_len, node_num, output_dim) - weekly average target
    target_day_avg: torch.Tensor       # (B, output_len, node_num, output_dim) - daily average target
    timestamp: torch.Tensor            # (B, window_size, 5)

    def __init__(self, history, history_week_avg, history_day_avg,
                 target, target_week_avg, target_day_avg,
                 timestamp, training=False) -> None:
        self.history = history
        self.history_week_avg = history_week_avg
        self.history_day_avg = history_day_avg
        self.target = target
        self.target_week_avg = target_week_avg
        self.target_day_avg = target_day_avg
        self.timestamp = timestamp
        self.training = training

    def __len__(self):
        return self.history.shape[0]
    
    def __getitem__(self, index):
        return (self.history[index],
                self.history_week_avg[index], self.history_day_avg[index],
                self.target[index],
                self.target_week_avg[index], self.target_day_avg[index],
                self.timestamp[index])


class DataProvider():
    node_num: int
    features: int
    data: torch.Tensor       # (T, node_num, features) - dữ liệu gốc
    timestamp: torch.Tensor  # (T, 5) - [month, day, weekday, hour, minute]
    steps_per_day: int = 288  # override in subclasses (e.g. 24 for 1h data)

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

    def _time_to_slot(self, hour, minute):
        """Convert (hour, minute) arrays to intra-day slot index [0, steps_per_day)."""
        return (hour * 60 + minute) * self.steps_per_day // (24 * 60)

    def compute_weekly_average(self, train_data, train_timestamp):
        spd = self.steps_per_day
        spw = spd * 7

        weekday = train_timestamp[:, 2].cpu().numpy()
        hour    = train_timestamp[:, 3].cpu().numpy()
        minute  = train_timestamp[:, 4].cpu().numpy()
        train_slot = weekday * spd + self._time_to_slot(hour, minute)  # (T_train,)

        full_weekday = self.timestamp[:, 2].cpu().numpy()
        full_hour    = self.timestamp[:, 3].cpu().numpy()
        full_minute  = self.timestamp[:, 4].cpu().numpy()
        full_slot = full_weekday * spd + self._time_to_slot(full_hour, full_minute)  # (T_full,)

        train_np = train_data.cpu().numpy().astype(np.float64)
        mask = (train_np != 0).astype(np.float64)

        lookup_sum = np.zeros((spw, self.node_num, self.features), dtype=np.float64)
        lookup_cnt = np.zeros((spw, self.node_num, self.features), dtype=np.float64)
        np.add.at(lookup_sum, train_slot, train_np * mask)
        np.add.at(lookup_cnt, train_slot, mask)

        lookup = (lookup_sum / np.maximum(lookup_cnt, 1)).astype(np.float32)
        return torch.from_numpy(lookup[full_slot])  # (T_full, N, F)

    def compute_daily_average(self, train_data, train_timestamp):
        spd = self.steps_per_day

        hour   = train_timestamp[:, 3].cpu().numpy()
        minute = train_timestamp[:, 4].cpu().numpy()
        train_slot = self._time_to_slot(hour, minute)  # (T_train,)

        full_hour   = self.timestamp[:, 3].cpu().numpy()
        full_minute = self.timestamp[:, 4].cpu().numpy()
        full_slot = self._time_to_slot(full_hour, full_minute)  # (T_full,)

        train_np = train_data.cpu().numpy().astype(np.float64)
        mask = (train_np != 0).astype(np.float64)

        lookup_sum = np.zeros((spd, self.node_num, self.features), dtype=np.float64)
        lookup_cnt = np.zeros((spd, self.node_num, self.features), dtype=np.float64)
        np.add.at(lookup_sum, train_slot, train_np * mask)
        np.add.at(lookup_cnt, train_slot, mask)

        lookup = (lookup_sum / np.maximum(lookup_cnt, 1)).astype(np.float32)
        return torch.from_numpy(lookup[full_slot])  # (T_full, N, F)

    def compute_monthly_average(self, train_data, train_timestamp):
        spd = self.steps_per_day
        spm = 31 * spd  # day-of-month 1-31, 0-indexed as (day-1)

        day    = train_timestamp[:, 1].cpu().numpy()  # 1-31
        hour   = train_timestamp[:, 3].cpu().numpy()
        minute = train_timestamp[:, 4].cpu().numpy()
        train_slot = (day - 1) * spd + self._time_to_slot(hour, minute)  # (T_train,)

        full_day    = self.timestamp[:, 1].cpu().numpy()
        full_hour   = self.timestamp[:, 3].cpu().numpy()
        full_minute = self.timestamp[:, 4].cpu().numpy()
        full_slot = (full_day - 1) * spd + self._time_to_slot(full_hour, full_minute)  # (T_full,)

        train_np = train_data.cpu().numpy().astype(np.float64)
        mask = (train_np != 0).astype(np.float64)

        lookup_sum = np.zeros((spm, self.node_num, self.features), dtype=np.float64)
        lookup_cnt = np.zeros((spm, self.node_num, self.features), dtype=np.float64)
        np.add.at(lookup_sum, train_slot, train_np * mask)
        np.add.at(lookup_cnt, train_slot, mask)

        lookup = (lookup_sum / np.maximum(lookup_cnt, 1)).astype(np.float32)
        return torch.from_numpy(lookup[full_slot])  # (T_full, N, F)

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
        
        # Compute averages from training data
        weekly_avg = self.compute_weekly_average(train_data, train_te).cuda()
        daily_avg = self.compute_daily_average(train_data, train_te).cuda()

        scaler_data = self.data[train_range[0]:train_range[1]]
        dim = scaler_data.shape[-1]
        mean = [scaler_data[..., i:i+1].mean() for i in range(dim)]
        std = [scaler_data[..., i:i+1].std() for i in range(dim)]
        self.scaler = self.getscalerclass()(mean, std)

        # Training set
        train_data_normal = self.data[train_range[0]:train_range[1]]
        train_week_avg = weekly_avg[train_range[0]:train_range[1]]
        train_day_avg = daily_avg[train_range[0]:train_range[1]]

        train_sample = generate_sample_by_sliding_window(train_data_normal, sample_len=window_size)
        train_sample_week = generate_sample_by_sliding_window(train_week_avg, sample_len=window_size)
        train_sample_day = generate_sample_by_sliding_window(train_day_avg, sample_len=window_size)

        train_x = train_sample[:, :sample_len, ..., :input_dim]
        train_x_week = train_sample_week[:, :sample_len, ..., :input_dim]
        train_x_day = train_sample_day[:, :sample_len, ..., :input_dim]
        train_y = train_sample[:, -output_len:, ..., :output_dim]
        train_y_week = train_sample_week[:, -output_len:, ..., :output_dim]
        train_y_day = train_sample_day[:, -output_len:, ..., :output_dim]

        train_x = self.scaler.transform(train_x)
        train_x_week = self.scaler.transform(train_x_week)
        train_x_day = self.scaler.transform(train_x_day)
        train_y = self.scaler.transform(train_y)
        train_y_week = self.scaler.transform(train_y_week)
        train_y_day = self.scaler.transform(train_y_day)

        train_te = generate_sample_by_sliding_window(train_te, sample_len=window_size)
        train_dataset = BasicDataset(
            history=train_x,
            history_week_avg=train_x_week, history_day_avg=train_x_day,
            target=train_y,
            target_week_avg=train_y_week, target_day_avg=train_y_day,
            timestamp=train_te, training=True
        )

        # Validation set
        val_data_normal = self.data[val_range[0]:val_range[1]]
        val_week_avg = weekly_avg[val_range[0]:val_range[1]]
        val_day_avg = daily_avg[val_range[0]:val_range[1]]
        val_te = self.timestamp[val_range[0]:val_range[1]]

        val_sample = generate_sample_by_sliding_window(val_data_normal, sample_len=window_size)
        val_sample_week = generate_sample_by_sliding_window(val_week_avg, sample_len=window_size)
        val_sample_day = generate_sample_by_sliding_window(val_day_avg, sample_len=window_size)

        val_x = val_sample[:, :sample_len, ..., :input_dim]
        val_x_week = val_sample_week[:, :sample_len, ..., :input_dim]
        val_x_day = val_sample_day[:, :sample_len, ..., :input_dim]
        val_y = val_sample[:, -output_len:, ..., :output_dim]
        val_y_week = val_sample_week[:, -output_len:, ..., :output_dim]
        val_y_day = val_sample_day[:, -output_len:, ..., :output_dim]

        val_x = self.scaler.transform(val_x)
        val_x_week = self.scaler.transform(val_x_week)
        val_x_day = self.scaler.transform(val_x_day)
        val_y = self.scaler.transform(val_y)
        val_y_week = self.scaler.transform(val_y_week)
        val_y_day = self.scaler.transform(val_y_day)

        val_te = generate_sample_by_sliding_window(val_te, sample_len=window_size)
        val_dataset = BasicDataset(
            history=val_x,
            history_week_avg=val_x_week, history_day_avg=val_x_day,
            target=val_y,
            target_week_avg=val_y_week, target_day_avg=val_y_day,
            timestamp=val_te
        )

        # Test set
        test_data_normal = self.data[test_range[0]:test_range[1]]
        test_week_avg = weekly_avg[test_range[0]:test_range[1]]
        test_day_avg = daily_avg[test_range[0]:test_range[1]]
        test_te = self.timestamp[test_range[0]:test_range[1]]

        test_sample = generate_sample_by_sliding_window(test_data_normal, sample_len=window_size)
        test_sample_week = generate_sample_by_sliding_window(test_week_avg, sample_len=window_size)
        test_sample_day = generate_sample_by_sliding_window(test_day_avg, sample_len=window_size)

        test_x = test_sample[:, :sample_len, ..., :input_dim]
        test_x_week = test_sample_week[:, :sample_len, ..., :input_dim]
        test_x_day = test_sample_day[:, :sample_len, ..., :input_dim]
        test_y = test_sample[:, -output_len:, ..., :output_dim]
        test_y_week = test_sample_week[:, -output_len:, ..., :output_dim]
        test_y_day = test_sample_day[:, -output_len:, ..., :output_dim]

        test_x = self.scaler.transform(test_x)
        test_x_week = self.scaler.transform(test_x_week)
        test_x_day = self.scaler.transform(test_x_day)
        test_y = self.scaler.transform(test_y)
        test_y_week = self.scaler.transform(test_y_week)
        test_y_day = self.scaler.transform(test_y_day)

        test_te = generate_sample_by_sliding_window(test_te, sample_len=window_size)
        test_dataset = BasicDataset(
            history=test_x,
            history_week_avg=test_x_week, history_day_avg=test_x_day,
            target=test_y,
            target_week_avg=test_y_week, target_day_avg=test_y_day,
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
    'ENERGY': lambda T: generatetimestamp(start='20190101 00:00:00', periods=T, freq='10min'),
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
    steps_per_day = 144  # 10-minute intervals (24*60/10)

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