import torch
from transformers import BertTokenizer, BertModel
import chinese_converter
import re
import numpy as np
from transformers import BertForTokenClassification
### from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def testing(text, model):

    data = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case = True)
    text_split = re.split('。', text)
    text_split_join = []
    temp = ''
    for i in text_split:
        if len(temp) + len(i) < 511:
            temp += i
        else:
            if len(temp) < 511:
                text_split_join.append(temp)
            temp = i
    if temp and len(temp) < 511:
        text_split_join.append(temp)
    for text in text_split_join:
        text = re.sub('[a-zA-Z0-9 ’!"#$%&\'()*+,-./:;<=>?@?★…【】《》？“”‘’！[\\]^_`{|}~]','',text)
        # print(len(text),text)
        text_convert = chinese_converter.to_simplified(text)
        encoded_text = tokenizer(text_convert, padding=True, return_tensors="pt", add_special_tokens=True)
        encoded_text = np.array(encoded_text["input_ids"])[0]
        # print(encoded_text)
        # predicted_index = encoded_text
        # predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in
        #            range(1, (len(encoded_text) - 1))]
        # print(predicted_token)
        outputs = model(input_ids=torch.tensor([encoded_text]).to('cuda:0'))
        logits = outputs[0][0][1:-1]
        # print(len(logits))
        index = 0
        for i in range(0, len(logits)):
            if logits[i][1] > logits[i][0]:
                if data and i == index + 1:
                    data[-1] += text[i]
                else:
                    data.append(text[i])
                index = i
    data = set(data)
    res = []
    for i in data:
        if 2 <= len(i) <= 4:
            res.append(i)
    print(res)
    return set(data)

def make_data(a, b):
    data = []
    model = BertModel.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case = True)
    data = []
    for i in range(a, b):
        f = open("training_set/" + str(i) + ".txt", "r")
        lines = f.readlines()

        label = lines[0]
        text = lines[1] if len(lines) == 2 else ''.join(lines[2:])
        f.close()
        text = re.sub('[a-zA-Z0-9 ’!"#$%&\'()*+,-./:;<=>?@?★…【】《》？“”‘’！[\\]^_`{|}~]','',text)
        text_split = re.split('。', text)
        text_split_join = []
        temp = ''
        for i in text_split:
            if len(temp) + len(i) < 511:
                temp += i
            else:
                if len(temp) < 511:
                    text_split_join.append(temp)
                temp = i
        if temp and len(temp) < 511:
            text_split_join.append(temp)

        for text in text_split_join:
            text = chinese_converter.to_simplified(text)

            encoded_text = tokenizer(text, padding=True, return_tensors="pt", add_special_tokens=True)#会自动添加ERNIE所需要的特殊token，如[CLS], [SEP]
            encoded_text = np.array(encoded_text["input_ids"])[0]#[:MAX_SEQLEN]
            #encoded_text = np.pad(encoded_text, (0, MAX_SEQLEN-len(encoded_text)), mode='constant') # 对所有句子都补长至11000，这样会比较费显存；

            if label[0:2] == '[]':
                label_cls = [0] * len(encoded_text)
            else:
                if type(label) == type(''):
                    label = label.split(', ')
                label_cls = [0] * len(encoded_text)
                for i in label:
                    i = i.split('\'')
                    i = chinese_converter.to_simplified(i[1])
                    encode_i = tokenizer(i, padding=True, return_tensors="pt", add_special_tokens=True)
                    encode_i = np.array(encode_i['input_ids'][0][1:-1])
                    for j in range(0, len(encoded_text)-len(encode_i)):
                        if (encoded_text[j:j+len(encode_i)] == encode_i).all():
                            for k in range(0, len(encode_i)):
                                label_cls[j+k] = 1
            data.append((torch.tensor(encoded_text), torch.tensor(np.array(label_cls))))
    return data


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    #segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    
    if samples[0][1] is not None:
        label_ids = [s[1] for s in samples]
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    label_ids = pad_sequence(label_ids, 
                                  batch_first=True)
#     segments_tensors = pad_sequence(segments_tensors, 
#                                     batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, masks_tensors, label_ids

def train(trainloader, validloader):
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 2

    # model = BertForTokenClassification.from_pretrained(
    #     PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    model = BertForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load('./model/0728_1_22_3.0149'))


    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    EPOCHS = 60
    for epoch in range(EPOCHS):
        
        # 訓練模式
        model.train()
        
        temp = 0
        running_loss = 0.0
        for data in trainloader:
            #print(temp)
            temp += 1
            tokens_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]

            # 將參數梯度歸零
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(input_ids=tokens_tensors,  
                            attention_mask=masks_tensors, 
                            labels=labels)
            logits = outputs[1]
            y_pred, y_true, masks_tensors, y_true_inverse = reshape_for_loss_fn(logits, labels, masks_tensors, device)

            loss = mask_BCE_loss(y_pred, y_true, masks_tensors, y_true_inverse, is_training = True)
            # print(outputs[0])
            # print(loss)
            #loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()

            # 紀錄當前 batch loss
            running_loss += loss.item()
        
        model.eval()

        for i in range(4972, 5000):
            f = open("training_set/" + str(i) + ".txt", "r")
            lines = f.readlines()
            text = lines[1] if len(lines) == 2 else ''.join(lines[2:])
            f.close()
            testing(text, model)

        valid_loss = 0
        for data in validloader:
            tokens_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]
            
            # forward pass

            outputs = model(input_ids=tokens_tensors,  
                            attention_mask=masks_tensors, 
                            labels=labels)
            logits = outputs[1]
            y_pred, y_true, masks_tensors, y_true_inverse = reshape_for_loss_fn(logits, labels, masks_tensors, device)

            valid_loss_batch =  mask_BCE_loss(y_pred, y_true, masks_tensors, y_true_inverse, is_training = False)
            valid_loss += valid_loss_batch.item()
            #valid_loss += outputs[0].item()
        # if valid_loss > temp_valid_loss and epoch > 3:
        #     break
        # else:
        #     valid_loss = temp_valid_loss
        print('valid_loss:',valid_loss)
        print('[epoch %d] loss: %.3f' %
              (epoch + 1, running_loss))
        torch.save(model.state_dict(), './model/0805_1_' + str(epoch + 1) + '_' + str(valid_loss)[:6] + '_' + str(running_loss)[:6])

def reshape_for_loss_fn(logits, labels, masks_tensors, device):
    #print(logits)
    m = torch.nn.Softmax(dim=2)
    logits = m(logits)

    y_pred = logits
    labels = torch.unsqueeze(labels, 2)
    y_true = torch.zeros(len(labels), len(labels[0]), len(labels[0][0])+1).to(device)
    #y_true = y_true.to(device)
    y_true_inverse = torch.zeros(len(labels), len(labels[0]), len(labels[0][0])+1).to(device)
    for i in range(len(labels[0])):
        for j in range(len(labels)):
            if labels[j][i][0].data == 1:
                y_true[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), labels[j][i]))
                y_true_inverse[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), (torch.zeros(1,dtype=torch.long)).to(device)))
            else:
                y_true[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long).to(device)), labels[j][i]))
                y_true_inverse[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), (torch.ones(1,dtype=torch.long)).to(device)))
            

            # if labels[j][i][0].data == 1:
            #     y_true[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), labels[j][i]*300))
            #     y_true_inverse[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), (torch.zeros(1,dtype=torch.long)).to(device)))
            # else:
            #     y_true[j][i] = torch.cat(((torch.ones(1,dtype=torch.long).to(device)), labels[j][i]))
            #     y_true_inverse[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), (torch.ones(1,dtype=torch.long)).to(device)))


    masks_tensors_s = torch.unsqueeze(masks_tensors, 2)
    masks_tensors = torch.zeros(len(masks_tensors_s), len(masks_tensors_s[0]), len(masks_tensors_s[0][0])+1)
    masks_tensors = masks_tensors.to(device)
    for i in range(len(masks_tensors_s[0])):
        for j in range(len(masks_tensors_s)):
            masks_tensors[j][i] = torch.cat((masks_tensors_s[j][i], masks_tensors_s[j][i]))
    return y_pred, y_true, masks_tensors, y_true_inverse

def mask_BCE_loss(y_pred, y_true, padding_mask, y_true_inverse, is_training=False, epsilon=1e-06):
    tp = (padding_mask * y_true * y_pred).sum().to(torch.float32)
    tn = (padding_mask * (y_true_inverse) * (1 - y_pred)).sum().to(torch.float32)
    fp = (padding_mask * (y_true_inverse) * y_pred).sum().to(torch.float32)
    fn = (padding_mask * y_true * (1 - y_pred)).sum().to(torch.float32)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    b = 1
    f1 = (1 + b) * (precision * recall) / (b * precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    # if is_training == False:
    #     print(tp,fp,fn)
    #     print(f1)
    return 1 - f1.mean()

    #return -1*torch.mean( padding_mask*y_true*torch.log(y_pred+epsilon) )

def main():
    trainset = make_data(1, 4000)
    validset = make_data(4000, 5024)
    #print(trainset)


    # 初始化一個每次回傳 64 個訓練樣本的 DataLoader
    # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
    BATCH_SIZE = 4

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                             collate_fn=create_mini_batch)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE, 
                             collate_fn=create_mini_batch)
    train(trainloader, validloader)

if __name__ == '__main__':
    main()
    # PRETRAINED_MODEL_NAME = "bert-base-chinese"
    # NUM_LABELS = 2
    # model = BertForTokenClassification.from_pretrained(
    #     PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    # for i in range(802, 803):
    #     f = open("training_set/" + str(i) + ".txt", "r")
    #     lines = f.readlines()
    #     text = lines[1] if len(lines) == 2 else lines[2]
    #     testing(text, model)


