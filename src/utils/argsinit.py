import argparse

def AddModelArgs(parser):

    parser.add_argument("--lora",action="store_true", help="whether use lora fine-tunning")

    parser.add_argument("--prompt_pool",action="store_true")

    parser.add_argument("--ln_grad",action="store_true", help="whether to calculate gradient of LayerNorm ")

    parser.add_argument("--causal", default=0, type=int,
                            help="LLM causal attention")
    
    parser.add_argument("--prompt_prefix", default=None ,type=str, help="whether use prompt or not")


    parser.add_argument("--node_embedding", action="store_true")

    parser.add_argument("--time_token", action="store_true")


    parser.add_argument("--model", default="gpt2" ,type=str)

    parser.add_argument("--llm_layers", default=None, type=int)

    parser.add_argument("--dropout", default=0, type=float)

    parser.add_argument("--trunc_k", default=16, type=int)

    parser.add_argument("--t_dim", default=64, type=int)

    parser.add_argument("--node_emb_dim", default=128, type=int)

    parser.add_argument("--sandglassAttn", type=int, default=0)
    parser.add_argument("--wo_conloss" , action="store_true")
    parser.add_argument("--sag_dim", default=128, type=int)
    parser.add_argument("--sag_tokens", default=128, type=int)
    # parser.add_argument("--use_anchor_diff_token", default=0, type=int, help="use_anchor_diff=1, use_anchor=2")
    # parser.add_argument("--use_diff", default=0, type=int, help="use_diff=1")
    # parser.add_argument("--use_sep_token", action="store_true")
    # parser.add_argument("--use_sep2_token", action="store_true")
    # parser.add_argument("--use_task_token", action="store_true", help="Enable task type token")
    # parser.add_argument("--use_context_token", action="store_true", help="Enable global context summary token")
    # parser.add_argument("--use_quality_token", action="store_true", help="Enable input quality assessment token")
    parser.add_argument("--use_anchor_week", action="store_true", help="use week anchor (weekly pattern)")
    parser.add_argument("--use_anchor_day", action="store_true", help="use day anchor (daily pattern)")
    parser.add_argument("--anchor_week_loss_weight", type=float, default=0.05, help="weight for week anchor deviation loss")
    parser.add_argument("--anchor_day_loss_weight", type=float, default=0.05, help="weight for day anchor deviation loss")
    parser.add_argument("--anchor_loss_type", type=str, default="huber", choices=["huber", "mae", "mse"], help="anchor loss type: huber or mae or mse")
    parser.add_argument("--use_instruction",action="store_true")

    # Ablation: bỏ một token trong 6 token của Time2TokenPerNode
    #   0=trend  1=season  2=residual  3=grad  4=stats  5=attn   -1=full (không bỏ)
    parser.add_argument("--ablation_drop_token", type=int, default=-1,
                        help="Drop one token from Time2TokenPerNode for ablation "
                             "(0=trend,1=season,2=residual,3=grad,4=stats,5=attn,-1=keep all)")
    parser.add_argument("--wo_conloss1", action="store_true", help="Ablation: remove sandglass attention consistency loss")
    parser.add_argument("--wo_conloss2", action="store_true", help="Ablation: remove Dirichlet regularization loss on attention distribution")


def AddDataArgs(parser):

    parser.add_argument("--dataset" ,type=str, default="pems04")

    parser.add_argument("--data_path" ,type=str)

    parser.add_argument("--adj_filename" ,default=None , type=str)

    parser.add_argument("--sample_len", default=12, type=int)

    parser.add_argument("--predict_len", default=12, type=int)

    # parser.add_argument("--node_num", type=int)

    # parser.add_argument("--features", type=int)

    parser.add_argument("--train_ratio", default=0.6, type=float)

    parser.add_argument("--val_ratio", default=0.6, type=float)

    parser.add_argument("--input_dim", default=1, type=int)

    parser.add_argument("--output_dim", default=1, type=int)




def AddTrainArgs(parser):

    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--lr_decay", default=0.99, type=float)

    parser.add_argument("--weight_decay", default=0.05, type=float)

    parser.add_argument("--batch_size", default=4, type=int)

    parser.add_argument("--epoch", default=100, type=int)

    parser.add_argument("--val_epoch", default=5, type=int)

    parser.add_argument("--test_epoch", default=5, type=int)

    parser.add_argument("--patience", default=100, type=int)


def InitArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--desc", default='phi2_s_token', type=str,
                            help="description")
    
    parser.add_argument("--log_root", default='../logs', type=str,
                            help="Log root directory")
    
    parser.add_argument("--from_pretrained_model" , default=None ,type=str)

    parser.add_argument("--zero_shot" , action="store_true")

    parser.add_argument("--nni" , action="store_true")

    parser.add_argument("--save_result" , action="store_true")

    parser.add_argument("--few_shot" , default=1, type=float)

    parser.add_argument("--node_shuffle_seed" , default=None, type=int)

    parser.add_argument("--trainset_dynamic_missing" , action="store_true")

    parser.add_argument("--task" , default='prediction' ,choices=['prediction','imputation','all'],type=str)
    
    parser.add_argument("--target_strategy" , default='random' ,choices=['random','hybrid'],type=str)

    AddDataArgs(parser)

    AddModelArgs(parser)

    AddTrainArgs(parser)

    args = parser.parse_args()

    return args