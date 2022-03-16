type='baseline'
learning_rate='2e-5'
epochs=4
batch_size=64
CUDA_VISIBLE_DEVICES=$1 python run.py --do_train \
--model_type "bert" \
--bert_model  "bert-base-uncased" \
--train_file "./data/quora/train" \
--dev_file "./data/quora/dev" \
--test_file "./data/quora/test" \
--name 'baseline'   \
--data 'quora'   \
--do_lower_case \
--learning_rate 2e-5 \
--epochs 4 \
--batch_size 64 \
--max_length 65 \
--fp16 --fptype O2 \
--save_dir "./save/$type/$epochs/$batch_size/$learning_rate/"

# nohup sh train.sh > ./logs/1.txt 2>&1 &