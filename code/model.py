import torch
from transformers import BertTokenizer, BertModel

eps = 1e-6
class BertClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_path='./pytorch-ernie'):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.linear_1 = torch.nn.Linear(768, 768)
        self.linear_2 = torch.nn.Linear(768, 1)

    def forward(self, x):
        embedding = self.bert(x)[0]
        embedding = embedding.view(-1, 768)
        output = self.linear_1(embedding)
        output = torch.relu(output)
        output = self.linear_2(output)
        output = output.view(x.size(0), -1)
        output = torch.sigmoid(output)
        return output

def mask_BCE_loss(y_pred, y_true, padding_mask, weight=1000):
    return -1*torch.mean( weight*padding_mask*y_true*torch.log(y_pred+eps)+ padding_mask*(1-y_true)*torch.log(1-y_pred+eps) )
