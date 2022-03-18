import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default="")
parser.add_argument('--dev_file', default="")
parser.add_argument('--test_file', default="")

parser.add_argument("--wandb_entity", type=str, default='rpeng')
parser.add_argument("--name", type=str, default="")
parser.add_argument('--data', type=str, default="")

parser.add_argument('--bert_model', default="bert-base-uncased")
parser.add_argument('--do_lower_case', action='store_true')
parser.add_argument("--checkpoint_epoch", default=-1, type=int)
parser.add_argument("--checkpoint", default=-1, type=int)
parser.add_argument('--save_dir', default="", type=str)

parser.add_argument("--save_step", default=100, type=int)

parser.add_argument('--model_type', default="model", type=str)
parser.add_argument('--seed', default=-1, type=int)

parser.add_argument('--fgm', action='store_true')
parser.add_argument('--rdrop', action='store_true')
parser.add_argument('--num_labels', default=2, type=int)

# TODO 常改动参数
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--adam_epsilon', default=1e-8, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)


parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=20, type=int) # 训练轮数
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_length', default=0, type=int)

parser.add_argument("--warmup_steps", default=0, type=int)

parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fptype', default="O1")

# test args
args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
args.device = device                                                                    