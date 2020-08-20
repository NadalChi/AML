## Installation
* Clone the repo to your local machine using https://github.com/NadalChi/AML.git


```python
!git clone https://github.com/NadalChi/AML.git
```

    Cloning into 'AML'...
    remote: Enumerating objects: 4914, done.
    remote: Counting objects: 100% (4914/4914), done.
    remote: Compressing objects: 100% (4795/4795), done.
    remote: Total 4914 (delta 71), reused 4887 (delta 53), pack-reused 0
    Receiving objects: 100% (4914/4914), 7.04 MiB | 747.00 KiB/s, done.
    Resolving deltas: 100% (71/71), done.
    Updating files: 100% (5036/5036), done.


- Install this package first.


```python
!pip install googledrivedownloader
```

    Collecting googledrivedownloader
      Using cached googledrivedownloader-0.4-py2.py3-none-any.whl (3.9 kB)
    Installing collected packages: googledrivedownloader
    Successfully installed googledrivedownloader-0.4


## Download model

* Run the following command. Or under folder AML/code, then run download_model.py


```python
%cd AML/code
%run -i 'download_model.py'
```

    /Users/huangchingchi/Desktop/AML/code/AML/code
     == Download model == 
    Downloading 1pC9OGFrsV4l_JCUT5Mn6eZgZXsDxFL8a into ../model/finetuned_token_cls_model... Done.
     == Done == 


## Load sample data
* Load sample data from data set.  
* You can choose any data you want between 1.txt to 5023.txt.  
* For example, if you want to load single data set 4800.txt, then you can call function as `aml.load_smaple_data(4800)`.  
* The implement result shows as below.


```python
import AML_readme as aml
label, text = aml.load_smaple_data(4801)
print(label, text)
label, text = aml.load_smaple_data(4800)
print(label, text)
```

    []
     40歲楊姓女子因缺錢花用，今年6月間帶著安姓小妹、陳姓小弟到清水區一家檳榔攤，楊女先持玩具槍喝令店員「不准動」，並由安女負責搜刮店內財物、陳男則負責把風，事後3人被捕，陳男雖辯稱「是被逼迫犯案」，仍被依強盜罪起訴。起訴指出，今年6月2日清晨，楊姓女子帶著陳姓小弟（39歲），先在彰化縣中正路附近強盜一部小轎車，之後由陳男負責駕車、載著楊女到台中市西屯區接安姓女子（26歲），接著3人共乘1車，沿途尋找犯案目標，行經清水區一家檳榔攤前時，便決定下手。手持玩具槍的楊女與安女先下車進入檳榔攤，由楊女持槍恐嚇店員「別動」，安女則進入拿走檳榔攤內現金共8300元，緊接著，楊女離去前還取走店內11包香菸，而整個過程，陳男都負責在車上把風、接應。警方事後獲報並逮獲3人，楊女坦承犯案，而安女、陳男都辯稱「是受楊女逼迫」，且陳男還供稱「事前因為有吃藥，根本記不得過程，只記得有開車、都是聽楊女指示」。台中檢方認為，楊女雖是主導，但陳男、安女明顯有自主意識，卻未趁機離開、還善盡把風之責，明顯是共犯，依強盜罪起訴3人。違反上述規定者，中時電子報有權刪除留言，或者直接封鎖帳號！請使用者在發言前，務必先閱讀留言板規則，謝謝配合。
    ['陳揚宗']
     （中央社記者劉世怡台北27日電）保安警察第二總隊刑事警察大隊小隊長陳揚宗被控利用查緝仿冒品案，詐騙知名法商路易威登馬爾悌耶公司（LV），二審今天依貪污等罪判刑2年、緩刑4年，須繳公庫15萬元。可上訴。檢方起訴指出，陳揚宗在民國104年11月、12月間查獲LV仿冒商品，明知查獲仿冒品後所需的貨車載運至贓物庫，運費均由公款支付，卻佯稱運費須由原告支付。受LV公司委託協助調查及鑑識工作的鑑定公司負責人，代為墊款運費新台幣2萬5000元。鑑定公司負責人再向LV公司請款獲准。檢方依貪污治罪條例利用職務詐取財物罪將陳揚宗起訴。一審法院考量他坦承犯行，無前科、有悔意，已繳回2萬5000元，且全案犯罪所得不到5萬元，判他2年徒刑、緩刑4年，須繳公庫15萬元。二審台灣高等法院今天宣判，依貪污治罪條例利用職務詐取財物罪二罪，判刑2年、緩刑4年，須繳公庫15萬元。可上訴。


## Process data
* As we have the sample data and the lables. We have to convert each word in sample data to token.  
* And give each word a lable. What we are doing is classification. If the word belongs to AML-related focal persons. We set the label to one else we set it to zero.  
* The implement result shows as below.


```python
tokens = aml.process_data(label, text)
print(tokens)
```

    [(tensor([ 101, 8020,  704, 1925, 4852, 6381, 5442, 1155,  686, 2592, 1378, 1266,
            3189, 4510, 8021,  924, 2128, 6356, 2175, 5018,  753, 2600, 7339, 1152,
             752, 6356, 2175, 1920, 7339, 2207, 7339, 7270, 7357, 2813, 2134, 6158,
            2971, 1164, 4500, 3389, 5351,  820, 1088, 1501, 3428, 8024, 6400, 7745,
            4761, 1399, 3791, 1555, 6662, 3211, 2014, 4633, 7716, 2209, 2635, 5456,
            1062, 1385, 8020, 8021, 8024,  753, 2144,  791, 1921,  898, 6576, 3738,
            5023, 5389, 1161, 1152, 2399,  510, 5353, 1152, 2399, 8024, 7557, 5373,
            1062, 2417,  674, 1039, 1377,  677, 6401, 3466, 3175, 6629, 6401, 2900,
            1139, 8024, 7357, 2813, 2134, 1762, 3696, 1744, 2399, 3299,  510, 3299,
            7313, 3389, 5815,  820, 1088, 1555, 1501, 8024, 3209, 4761, 3389, 5815,
             820, 1088, 1501, 1400, 2792, 7444, 4638, 6573, 6756, 6770, 6817, 5635,
            6597, 4289, 2417, 8024, 6817, 6589, 1772, 4507, 1062, 3621, 3118,  802,
            8024, 1316,  879, 4917, 6817, 6589, 7557, 4507, 1333, 1440, 3118,  802,
            1358, 1062, 1385, 1999, 2805, 1291, 1221, 6444, 3389, 1350, 7063, 6399,
            2339,  868, 4638, 7063, 2137, 1062, 1385, 6566, 6569,  782, 8024,  807,
             711, 1807, 3621, 6817, 6589, 3173, 1378, 2355,  674, 1039, 7063, 2137,
            1062, 1385, 6566, 6569,  782, 1086, 1403, 1062, 1385, 6435, 3621, 5815,
            1114, 3466, 3175,  898, 6576, 3738, 3780, 5389, 3340,  891, 1164, 4500,
            5466, 1218, 6400, 1357, 6568, 4289, 5389, 2199, 7357, 2813, 2134, 6629,
            6401,  671, 2144, 3791, 7368, 5440, 7030,  800, 1788, 2824, 4306, 6121,
            8024, 3187, 1184, 4906,  510, 3300, 2637, 2692, 8024, 2347, 5373, 1726,
             674, 1039, 8024,  684, 1059, 3428, 4306, 5389, 2792, 2533,  679, 1168,
             674, 1039, 8024, 1161,  800, 2399, 2530, 1152,  510, 5353, 1152, 2399,
            8024, 7557, 5373, 1062, 2417,  674, 1039,  753, 2144, 1378, 3968, 7770,
            5023, 3791, 7368,  791, 1921, 2146, 1161, 8024,  898, 6576, 3738, 3780,
            5389, 3340,  891, 1164, 4500, 5466, 1218, 6400, 1357, 6568, 4289, 5389,
             753, 5389, 8024, 1161, 1152, 2399,  510, 5353, 1152, 2399, 8024, 7557,
            5373, 1062, 2417,  674, 1039, 1377,  677, 6401,  102]), tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]


## Train model
* Create training set and validation set.  
* The model will saved under folder `'~/model'`.


```python
trainset = aml.load_processed_data(1, 4000)
validset = aml.load_processed_data(4000, 5024)
BATCH_SIZE = 4

trainloader = aml.DataLoader(trainset, batch_size=BATCH_SIZE, 
                         collate_fn=aml.create_mini_batch)
validloader = aml.DataLoader(validset, batch_size=BATCH_SIZE, 
                         collate_fn=aml.create_mini_batch)
EPOCHS = 30
aml.train(trainloader, validloader, EPOCHS)
```

## Load model
* The model we provide was trained by using 1.txt to 4000.txt from data set.  
* Load the model we downloaded at begin. The path was set equal to `'../model/finetuned_token_cls_model'`.  
* You can also load the path where your model at.  


```python
model = aml.load_model()
```

    Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertForTokenClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-chinese and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


## Inference
* If you are doing inference. First argument is the article you are going to inference. Second argument is the model you loaded.  
* The inference result shows as below.


```python
text = aml.load_smaple_data(4801)[1]
aml.inference(text, model)
```




    []




```python
text = aml.load_smaple_data(4802)[1]
aml.inference(text, model)
```




    ['徐詩彥', '林繼蘇']


