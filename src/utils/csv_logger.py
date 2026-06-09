import csv
import os
from datetime import datetime


def log_result_to_csv(args, mae, rmse, mape, total_params, total_trainable_params, csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(args.log_root, 'experiment_results.csv')

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    columns = [
        # Meta
        'timestamp',
        'desc',
        'dataset',
        'task',

        # Model params
        'model',
        'llm_layers',   
        'node_emb_dim',
        't_dim',
        'sag_dim',
        'sag_tokens',
        'trunc_k',
        'dropout',

        # Data params
        'sample_len',
        'predict_len',
        'input_dim',
        'output_dim',
        'train_ratio',
        'val_ratio',
        'few_shot',

        # Training params
        'lr',
        'weight_decay',
        'batch_size',
        'epoch',
        'patience',

        # Boolean flags (on/off status)
        'lora',
        'ln_grad',
        'node_embedding',
        'time_token',
        'sandglassAttn',
        'use_instruction',
        'use_anchor_day',
        'use_anchor_week',
        'wo_conloss',
        'zero_shot',

        # Anchor loss weights
        'anchor_day_loss_weight',
        'anchor_week_loss_weight',
        'anchor_loss_type',

        # Model size
        'total_params',
        'total_trainable_params',

        # Results (3 losses)
        'MAE',
        'RMSE',
        'MAPE',
    ]

    # Build the row data
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'desc': getattr(args, 'desc', ''),
        'dataset': getattr(args, 'dataset', ''),
        'task': getattr(args, 'task', ''),

        'model': getattr(args, 'model', ''),
        'llm_layers': getattr(args, 'llm_layers', ''),
        't_dim': getattr(args, 't_dim', ''),
        'node_emb_dim': getattr(args, 'node_emb_dim', ''),
        'sag_dim': getattr(args, 'sag_dim', ''),
        'sag_tokens': getattr(args, 'sag_tokens', ''),
        'trunc_k': getattr(args, 'trunc_k', ''),
        'dropout': getattr(args, 'dropout', ''),

        'sample_len': getattr(args, 'sample_len', ''),
        'predict_len': getattr(args, 'predict_len', ''),
        'input_dim': getattr(args, 'input_dim', ''),
        'output_dim': getattr(args, 'output_dim', ''),
        'train_ratio': getattr(args, 'train_ratio', ''),
        'val_ratio': getattr(args, 'val_ratio', ''),
        'few_shot': getattr(args, 'few_shot', ''),

        'lr': getattr(args, 'lr', ''),
        'weight_decay': getattr(args, 'weight_decay', ''),
        'batch_size': getattr(args, 'batch_size', ''),
        'epoch': getattr(args, 'epoch', ''),
        'patience': getattr(args, 'patience', ''),

        'lora': 'ON' if getattr(args, 'lora', False) else 'OFF',
        'ln_grad': 'ON' if getattr(args, 'ln_grad', False) else 'OFF',
        'node_embedding': 'ON' if getattr(args, 'node_embedding', False) else 'OFF',
        'time_token': 'ON' if getattr(args, 'time_token', False) else 'OFF',
        'sandglassAttn': getattr(args, 'sandglassAttn', 0),
        'use_instruction': 'ON' if getattr(args, 'use_instruction', False) else 'OFF',
        'use_anchor_day': 'ON' if getattr(args, 'use_anchor_day', False) else 'OFF',
        'use_anchor_week': 'ON' if getattr(args, 'use_anchor_week', False) else 'OFF',
        'wo_conloss': 'ON' if getattr(args, 'wo_conloss', False) else 'OFF',
        'zero_shot': 'ON' if getattr(args, 'zero_shot', False) else 'OFF',

        'anchor_day_loss_weight': getattr(args, 'anchor_day_loss_weight', ''),
        'anchor_week_loss_weight': getattr(args, 'anchor_week_loss_weight', ''),
        'anchor_loss_type': getattr(args, 'anchor_loss_type', 'huber'),

        'total_params': total_params,
        'total_trainable_params': total_trainable_params,

        'MAE': f'{mae:.4f}' if mae is not None else '',
        'RMSE': f'{rmse:.4f}' if rmse is not None else '',
        'MAPE': f'{mape:.4f}' if mape is not None else '',
    }

    # Check if file exists to decide whether to write headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return csv_path
