CUDA_VISIBLE_DEVICES=0 python3 run.py --do_train \
--get_badcase   \
--model_type "bert" \
--bert_model  "../ckpts/bert-base-uncased/" \
--train_file "./data/quora/train" \
--dev_file "./data/quora/dev" \
--test_file "./data/quora/test" \
--name 'bert-grdrop'   \
--data 'quora'   \
--do_lower_case \
--learning_rate 2e-5 \
--epochs 4 \
--batch_size 64 \
--max_length 65 \
--fp16 --fptype O2 \
--alpha 0.1 \
--save_dir "./save/"

# nohup sh train.sh > ./logs/1.txt 2>&1 &