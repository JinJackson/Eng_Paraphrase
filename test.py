predict_file = r'save/mrpc/test_save/res'
with open(predict_file, 'r', encoding='utf-8') as reader:
    all_predict = reader.readlines()
    all_predict = [int(x.strip()) for x in all_predict]
    print(all_predict)