
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from dataset import TrainDataBert
import argparse
import torch
from tqdm import tqdm
import numpy as np
from utils.metrics import acc_eval
import time
import os


models = {
  'bert': BertForSequenceClassification,
  'roberta': RobertaForSequenceClassification
}

tokenizers = {
  'bert': BertTokenizer,
  'roberta': RobertaTokenizer
}

def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]

def get_args():
    parser_test = argparse.ArgumentParser()

    parser_test.add_argument('--test_file', default='quora', type=str)
    # parser_test.add_argument('--model_type', default='vae', type=str, required=True) #vae baseline multi-task
    parser_test.add_argument('--save_dir', default=r'save/quora/baseline_1_10000', type=str) # test_model dir
    # parser_test.add_argument('--save_dir', default=r'save/mrpc/ck_4_1100', type=str)
    parser_test.add_argument('--max_length', default=128, type=int)
    parser_test.add_argument('--batch_size', default=64, type=int)
    args_test = parser_test.parse_args()
    return args_test
    
    

args_test = get_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

localtime = time.localtime(time.time())#获取当前时间
time = time.strftime('%Y%m%d',time.localtime(time.time()))#把获取的时间转换成"年月日格式”

output_dir = r'./save/' + args_test.test_file + '/test_save/' + time + args_test.save_dir.split('/')[-1]
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

test_file = './data/' + args_test.test_file + '/dev'

def get_wrong_case(all_labels):
    wrong_case = []
    all_labels = _to_list(np.squeeze(all_labels).tolist())
    all_labels = np.argmax(all_labels, axis = 1)
    print(all_labels)
    predict_file = output_dir + '/res'
    with open(predict_file, 'r', encoding='utf-8') as reader:
        all_predict = reader.readlines()
        all_predict = [int(x.strip()) for x in all_predict]
    
    with open(test_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
    print(len(all_predict) == len(all_labels), len(all_predict) == len(lines))

    for i in range(len(all_predict)):
        if all_predict[i] != all_labels[i]:
            wrong_case.append(lines[i])
    
    with open(output_dir + '/wrongcase.txt', 'w', encoding='utf-8') as writer:
      for case in wrong_case:
        writer.write(case)

    
        


def test(model, test_dataset, tokenizer):

    

    train_data = TrainDataBert(train_file=test_file, 
                               max_length=args_test.max_length,
                               tokenizer=tokenizer)
    train_dataLoader = DataLoader(dataset=train_data,
                                batch_size=args_test.batch_size,
                                shuffle=False)
    loss = []

    all_labels = None
    all_logits = None
    model.eval()
    model.to(device)

    softmax = torch.nn.Softmax(dim=1)

    for batch in tqdm(train_dataLoader, desc='test', ncols=50): 
        query1, query2 = batch[-2:]
        batch = tuple(t.to(device) for t in batch[:-2])
        input_ids, token_type_ids, attention_mask, labels = batch


        
        with torch.no_grad():
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            attention_mask=attention_mask.long(),
                            labels=labels.long(),
                            return_dict=True)

            eval_loss, logits = outputs['loss'], outputs['logits']
            loss.append(eval_loss.item())
            logits = softmax(logits)

            labels_ = []
            for label in labels:
              l = [0] * 2
              l[label] = 1
              labels_.append(l)

            labels_ = torch.from_numpy(np.array(labels_)).float().cuda()

            if all_labels is None:
                all_labels = labels_.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, labels_.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
    
    start = 0
    acc = acc_eval(all_labels, all_logits)
    if output_dir is not None:
      f = open(output_dir + '/res', 'w', encoding='utf-8')
      fs = open(output_dir + '/score', 'w', encoding='utf-8')
      for slist in all_logits:
        if slist[1] >= 0.5:
          f.write('1\n')
        else:
          f.write('0\n')
        fs.write(str(slist[1]) + '\n')
      f.close()
      fs.close()

    return np.array(loss).mean(), acc, all_labels


if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained(args_test.save_dir)
    tokenizer = BertTokenizer.from_pretrained(args_test.save_dir)
    loss, acc, all_labels = test(model, test_file, tokenizer)
    print(type(all_labels))
    get_wrong_case(all_labels)
    print('loss:', loss)
    print('acc', acc)
        