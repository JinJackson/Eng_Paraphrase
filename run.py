#coding=utf-8
from parser import args
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
import os, random, shutil
import glob
import torch
import wandb
import numpy as np
from tqdm import tqdm, trange
from pathlib import Path

from dataset import TrainDataBert, EvalDataBert
import transformers
transformers.logging.set_verbosity_error()

from utils.metrics import acc_eval
from utils.logger import getLogger
import torch.nn.functional as F

if args.fgm:
  from utils.FGM import FGM


models = {
  'bert': BertForSequenceClassification,
  'roberta': RobertaForSequenceClassification
}

tokenizers = {
  'bert': BertTokenizer,
  'roberta': RobertaTokenizer
}

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

if args.seed > -1:
  seed_torch(args.seed)


logger = None

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task ori:sum
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def train(model, tokenizer, checkpoint, out_path):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None
    # 训练数据处理
    train_data = TrainDataBert(train_file=args.train_file, 
                               max_length=args.max_length,
                               tokenizer=tokenizer)
    train_dataLoader = DataLoader(dataset=train_data,
                                batch_size=args.batch_size,
                                shuffle=True)


    # 初始化 optimizer，scheduler
    t_total = len(train_dataLoader) * args.epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
     # apex
    if args.fp16:
      model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    # 读取断点 optimizer、scheduler
    checkpoint_dir = out_path + "/checkpoint-" + str(checkpoint)
    if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        if args.fp16:
          amp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "amp.pt")))

    # 开始训练
    logger.debug("***** Running training *****")
    logger.debug("  Num examples = %d", len(train_dataLoader))
    logger.debug("  Num Epochs = %d", args.epochs)
    logger.debug("  Batch size = %d", args.batch_size)

    # 没有历史断点，则从0开始
    if checkpoint < 0:
        checkpoint = 0
    else:
        checkpoint += 1
    logger.debug("  Start Batch = %d", checkpoint)
    max_acc = 0.0
    cur_batch = 0

    # if args.fgm or args.grdrop or args.grdrop_logits:
    #   fgm = FGM(model)
    # t = torch.nn.Parameter(torch.FloatTensor(1)).to(args.device)
    for epoch in range(checkpoint, args.epochs):
        model.train()
        epoch_loss = []
        
        for batch in tqdm(train_dataLoader, desc="Iteration", ncols=50):
            cur_batch += 1
            model.zero_grad()
            # 设置tensor gpu运行
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids=input_ids,
                            token_type_ids= None if args.model_type == 'roberta' else token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels, output_hidden_states = True, return_dict = True)
            # cls
            loss, logits, cls = outputs['loss'],  outputs['logits'], outputs['hidden_states'][-1][:, 0, :]
            
            """ 第一次反向传播 """
            if not args.rdrop:
              if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
              else:
                loss.backward()
              if not args.fgm:
                epoch_loss.append(loss.item())

            # """ R-Drop 策略 """
            # if args.rdrop:
            #   outputs_rd = model(input_ids=input_ids,
            #                   token_type_ids= None if args.model_type == 'roberta' else token_type_ids,
            #                   attention_mask=attention_mask,
            #                   labels=labels)
            #   loss_rd, logits_rd = outputs_rd['loss'], outputs_rd['logits']
            #   ce_loss = 0.5 * (loss + loss_rd)
            #   kl_loss = compute_kl_loss(logits, logits_rd)
            #   final_loss = ce_loss + kl_loss
            #   if args.fp16:
            #     with amp.scale_loss(final_loss, optimizer) as rd_scaled_loss:
            #         rd_scaled_loss.backward()
            #   else:
            #     final_loss.backward()
            #   epoch_loss.append(final_loss.item())

            # """ FGM 策略 """
            # if args.fgm:
            #   fgm.attack()   #根据梯度进行扰动
            #   outputs_adv = model(input_ids=input_ids,
            #                   token_type_ids= None if args.model_type == 'roberta' else token_type_ids,
            #                   attention_mask=attention_mask,
            #                   labels=labels)
            #   loss_adv = outputs_adv['loss']
            #   if args.fp16:
            #     with amp.scale_loss(loss_adv, optimizer) as scaled_loss_adv:
            #         scaled_loss_adv.backward()
            #   else:
            #     loss_adv.backward()
            #   fgm.restore()  #恢复参数
            #   epoch_loss.append(loss.item() + loss_adv.item())


            optimizer.step()
            scheduler.step()   
           
            if cur_batch % args.save_step == 0 and epoch > 0:
              output_dir = out_path + "/checkpoint-" + str(epoch) + '-' + str(cur_batch)
              # eval dev
              eval_loss, eval_acc = evaluate(model, tokenizer, eval_file=args.dev_file, checkpoint=epoch)
              logger.debug('【DEV】Train Epoch %d: train_loss=%.4f, acc=%.4f' % (epoch, np.array(epoch_loss).mean(), eval_acc))
              # logger.info('***************t=%.4f*****************' % (t))
              wandb.log(
                    {
                        "train loss": np.array(epoch_loss).mean(),
                        "valid acc": eval_acc,
                    }, step=cur_batch,
              )
              if max_acc < eval_acc:
                max_acc = eval_acc
                wandb.run.summary["best val acc"] = max_acc
                # 输出日志 + 保存日志
                logger.info('【DEV】Train Epoch %d: train_loss=%.4f, acc=%.4f' % (epoch, np.array(epoch_loss).mean(), eval_acc))
                
                # eval test
                if args.test_file:
                  test_eval_loss, test_acc = evaluate(model, tokenizer, eval_file=args.test_file, checkpoint=epoch)
                  logger.info('【TEST】Train Epoch %d: train_loss=%.4f, acc=%.4f' % (epoch, np.array(epoch_loss).mean(), test_acc))
                  wandb.log(
                    {
                        "test acc": test_acc,
                    }, step=cur_batch,
                  )
                
                # # 删除历史模型
                # filelist = os.listdir(out_path)
                # for f in filelist:
                #   filepath = os.path.join(out_path, f)
                #   if os.path.isdir(filepath):
                #     shutil.rmtree(filepath,True)

                # 保存模型
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (model.module if hasattr(model, "module") else model)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.debug("Saving model checkpoint to %s", output_dir)
                if args.fp16:
                  torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.debug("Saving optimizer and scheduler states to %s", output_dir)
    logger.info('Training Finished')

              

def evaluate(model, tokenizer, eval_file, checkpoint, output_dir=None):
    # eval数据处理: eval可能是test或者dev
    eval_data = EvalDataBert(eval_file=eval_file,
                             max_length=args.max_length,
                             tokenizer=tokenizer)
    
    eval_dataLoader = DataLoader(dataset=eval_data,
                                    batch_size=args.batch_size,
                                    shuffle=False)
    

    logger.debug("***** Running evaluation {} *****".format(checkpoint))
    logger.debug("  Num examples = %d", len(eval_dataLoader))
    logger.debug("  Batch size = %d", args.batch_size)

    loss = []

    all_labels = None
    all_logits = None
    model.eval()

    softmax = torch.nn.Softmax(dim=1)

    for batch in tqdm(eval_dataLoader, desc='test', ncols=50): 
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, token_type_ids, attention_mask, labels = batch


        
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            token_type_ids=None if args.model_type == 'roberta' else token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True)

            eval_loss, logits = outputs['loss'], outputs['logits']
            loss.append(eval_loss.item())
            logits = softmax(logits)

            labels_ = []
            for label in labels:
              l = [0] * args.num_labels
              l[label] = 1
              labels_.append(l)

            # labels_ = []
            # for label in labels:
            #   if label == 0:
            #     labels_.append([1, 0])
            #   else:
            #     labels_.append([0, 1])
            labels_ = torch.from_numpy(np.array(labels_)).float().cuda()

            if all_labels is None:
                all_labels = labels_.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, labels_.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
    
    # 评价指标
    start = 0
    acc = acc_eval(all_labels, all_logits)
    if output_dir is not None:
      f = open(output_dir + '/res', 'w')
      fs = open(output_dir + '/score', 'w')
      for slist in all_logits:
        if slist[1] >= 0.5:
          f.write('1\n')
        else:
          f.write('0\n')
        fs.write(str(slist[1]) + '\n')
      f.close()
      fs.close()

    return np.array(loss).mean(), acc


if __name__ == "__main__":

    # 创建存储目录
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    out_path = args.save_dir+f"{args.name}-{args.data}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    logger = getLogger(__name__, os.path.join(out_path,'log.txt'))

    
    os.environ["WANDB_MODE"] = "offline"
    if args.do_train:
        wandb.init(
            project=f"{args.name}-{args.data}", config=args, entity=args.wandb_entity
        )
        # train： 接着未训练完checkpoint继续训练
        checkpoint = -1
        for checkpoint_dir_name in glob.glob(out_path + "/*"):
            try:
                checkpoint = max(checkpoint, int(checkpoint_dir_name.split('/')[-1].split('-')[1]))
            except Exception as e:
                pass
        checkpoint_dir = out_path + "/checkpoint-" + str(checkpoint)
        if checkpoint > -1:
          logger.debug(f" Load Model from {checkpoint_dir}")

        tokenizer = tokenizers[args.model_type].from_pretrained(args.bert_model if checkpoint == -1 else checkpoint_dir,
                                                do_lower_case=args.do_lower_case)
        
        model = models[args.model_type].from_pretrained(args.bert_model if checkpoint == -1 else checkpoint_dir, num_labels=args.num_labels) # default 2
        model.to(args.device)
        # 训练
        train(model, tokenizer, checkpoint, out_path)

    else:
        # eval：指定模型
        checkpoint = args.checkpoint
        checkpoint_epoch = args.checkpoint_epoch
        checkpoint_dir = out_path + "/checkpoint-" + str(checkpoint_epoch) + '-' + str(checkpoint)

        tokenizer = tokenizers[args.model_type].from_pretrained(checkpoint_dir,
                                                do_lower_case=args.do_lower_case)
        model = models[args.model_type].from_pretrained(checkpoint_dir, num_labels=args.num_labels)
        model.to(args.device)
        # 评估
        eval_loss, eval_acc = evaluate(model, tokenizer, eval_file=args.test_file, checkpoint=checkpoint, output_dir=checkpoint_dir)
        logger.debug('Evaluate Epoch %d: acc=%.4f' % (checkpoint, eval_acc))
