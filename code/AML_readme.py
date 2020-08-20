import torch
from transformers import BertTokenizer, BertModel
import chinese_converter
import re
import numpy as np
from transformers import BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def load_smaple_data(num):
	f = open("../data/training_set/" + str(num) + ".txt", "r")
	lines = f.readlines()

	label = lines[0]
	text = lines[1] if len(lines) == 2 else ''.join(lines[2:])
	f.close()
	return label, text

def process_data(label, text):
	data = []
	tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case = True)
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

		encoded_text = tokenizer(text, padding=True, return_tensors="pt", add_special_tokens=True)
		encoded_text = np.array(encoded_text["input_ids"])[0]

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
def load_processed_data(num, num1):
	processed_data = []
	for i in range(num, num1):
		label, text = load_smaple_data(i)
		one_data = process_data(label, text)
		for j in one_data:
			processed_data.append(j)
	return processed_data
def create_mini_batch(samples):
	tokens_tensors = [s[0] for s in samples]

	if samples[0][1] is not None:
		label_ids = [s[1] for s in samples]
	else:
		label_ids = None
	
	tokens_tensors = pad_sequence(tokens_tensors, 
								  batch_first=True)
	label_ids = pad_sequence(label_ids, 
								  batch_first=True)

	masks_tensors = torch.zeros(tokens_tensors.shape, 
								dtype=torch.long)
	masks_tensors = masks_tensors.masked_fill(
		tokens_tensors != 0, 1)
	
	return tokens_tensors, masks_tensors, label_ids

def train(trainloader, validloader, EPOCHS):
	PRETRAINED_MODEL_NAME = "bert-base-chinese"
	NUM_LABELS = 2

	model = BertForTokenClassification.from_pretrained(
		PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)


	# 使用 Adam Optim 更新整個分類模型的參數
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	for epoch in range(EPOCHS):
		
		# 訓練模式
		model.train()
		
		temp = 0
		running_loss = 0.0
		for data in trainloader:
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

			loss.backward()
			optimizer.step()

			# 紀錄當前 batch loss
			running_loss += loss.item()
		
		model.eval()

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
		
		print('valid_loss:',valid_loss)
		print('[epoch %d] loss: %.3f' %
			  (epoch + 1, running_loss))
		torch.save(model.state_dict(), '../model/0820_1_' + str(epoch + 1) + '_' + str(valid_loss)[:6] + '_' + str(running_loss)[:6])

def reshape_for_loss_fn(logits, labels, masks_tensors, device):
	m = torch.nn.Softmax(dim=2)
	logits = m(logits)

	y_pred = logits
	labels = torch.unsqueeze(labels, 2)
	y_true = torch.zeros(len(labels), len(labels[0]), len(labels[0][0])+1).to(device)
	y_true_inverse = torch.zeros(len(labels), len(labels[0]), len(labels[0][0])+1).to(device)
	for i in range(len(labels[0])):
		for j in range(len(labels)):
			if labels[j][i][0].data == 1:
				y_true[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), labels[j][i]))
				y_true_inverse[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), (torch.zeros(1,dtype=torch.long)).to(device)))
			else:
				y_true[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long).to(device)), labels[j][i]))
				y_true_inverse[j][i] = torch.cat(((torch.zeros(1,dtype=torch.long)).to(device), (torch.ones(1,dtype=torch.long)).to(device)))

	masks_tensors_s = torch.unsqueeze(masks_tensors, 2)
	masks_tensors = torch.zeros(len(masks_tensors_s), len(masks_tensors_s[0]), len(masks_tensors_s[0][0])+1)
	masks_tensors = masks_tensors.to(device)
	for i in range(len(masks_tensors_s[0])):
		for j in range(len(masks_tensors_s)):
			masks_tensors[j][i] = torch.cat((masks_tensors_s[j][i], masks_tensors_s[j][i]))
	return y_pred, y_true, masks_tensors, y_true_inverse

def mask_BCE_loss(y_pred, y_true, padding_mask, y_true_inverse, is_training=False, epsilon=1e-06):
	return -1*torch.mean( padding_mask*y_true*torch.log(y_pred+epsilon) )

def load_model(path='../model/finetuned_token_cls_model'):
	PRETRAINED_MODEL_NAME = "bert-base-chinese"
	NUM_LABELS = 2

	model = BertForTokenClassification.from_pretrained(
		PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.load_state_dict(torch.load(path, map_location=torch.device(device)))
	model.eval()
	return model

def inference(text, model):

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

		text_convert = chinese_converter.to_simplified(text)
		encoded_text = tokenizer(text_convert, padding=True, return_tensors="pt", add_special_tokens=True)
		encoded_text = np.array(encoded_text["input_ids"])[0]
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		outputs = model(input_ids=torch.tensor([encoded_text]).to((device)))
		logits = outputs[0][0][1:-1]

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
	return res





