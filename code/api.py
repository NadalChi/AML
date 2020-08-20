
from flask import Flask
from flask import request
from flask import jsonify
import datetime
import hashlib
import numpy as np
import pandas as pd

import torch
from transformers import BertTokenizer, BertModel
import chinese_converter
import re
from transformers import BertForTokenClassification

app = Flask(__name__)
####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'EMAIL@gmail.com'       #
SALT = 'qwert'                          #
#########################################

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

        text_convert = chinese_converter.to_simplified(text)
        encoded_text = tokenizer(text_convert, padding=True, return_tensors="pt", add_special_tokens=True)
        encoded_text = np.array(encoded_text["input_ids"])[0]
        outputs = model(input_ids=torch.tensor([encoded_text]).to(("cuda:0")))
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
def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def predict(article):
    """ Predict your model result
    @param article (str): a news article
    @returns prediction (list): a list of name
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######

    prediction = testing(article, model)

    #prediction = ['aha','danny','jack']
    
    
    ####################################################
    prediction = _check_datatype_to_list(prediction)
    return prediction

def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.
        
    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)
      
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """

    data = request.get_json(force=True)  

    esun_timestamp = data['esun_timestamp'] #自行取用
    
    t = datetime.datetime.now()  
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    
    try:
        answer = predict(data['news'])
    except:
        raise ValueError('Model error.')        
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})

if __name__ == "__main__":
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 2

    model = BertForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load('../model/finetuned_token_cls_model'))
    model = model.to("cuda:0")
    model.eval()

    app.run(host='0.0.0.0', port=80, debug=True)
