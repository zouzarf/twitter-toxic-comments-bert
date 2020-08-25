import pandas as pd
from plotly import tools
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib
from tqdm import tqdm
from statistics import *
import plotly.express as px
from plotly.subplots import make_subplots
import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import json
from statistics import *
from sklearn.feature_extraction.text import CountVectorizer
import os
import gc
import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import BertTokenizer,BertConfig,TFBertModel
from tqdm import tqdm
tqdm.pandas()
nltk.download('stopwords')
stop=set(stopwords.words('english'))


def read_train():
    train=pd.read_csv("./train.csv")
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    return train

def read_test():
    test=pd.read_csv("./test.csv")
    test['text']=test['text'].astype(str)
    return test

def read_submission():
    test=pd.read_csv("./sample_submission.csv")
    return test

#clean stop words url html emoji punctuation multiple spaces
def clean_df(df, train=True):
    df["dirty_text"] = df['text']
    def remove_stopwords(text):
        if text is not None:
            tokens = [x for x in word_tokenize(text) if x not in stop]
            return " ".join(tokens)
        else:
            return None
    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)    
    def remove_html(text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)
    # Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    
    def remove_punct(text):
        table=str.maketrans('','',string.punctuation)
        return text.translate(table)
    
    df["text"] = df['text'].apply(lambda x : x.lower())
    df['text']=df['text'].apply(lambda x: remove_emoji(x)) 
    df['text']=df['text'].apply(lambda x : remove_URL(x))
    df['text']=df['text'].apply(lambda x : remove_html(x)) 
    df['text'] =df['text'].apply(lambda x : remove_stopwords(x)) 
    df['text']=df['text'].apply(lambda x : remove_punct(x))
    df.text = df.text.replace('\s+', ' ', regex=True)
    if train:
        df["selected_text"] = df['selected_text'].apply(lambda x : x.lower())
        df['selected_text']=df['selected_text'].apply(lambda x: remove_emoji(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_URL(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_html(x))
        df['selected_text'] =df['selected_text'].apply(lambda x : remove_stopwords(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_punct(x))
        df.selected_text = df.selected_text.replace('\s+', ' ', regex=True)
    return df

def whole_text_classifier(test):
    test["selected_text"] = test['text']
    test = test[["textID", "selected_text"]]
    test.to_csv('whole_text_submission.csv',index=False)

def find_offset(x,y):
    """find offset in fail scenarios (only handles the start fail as of now)"""
    x_str = ' '.join(x)
    y_str = ' '.join(y)
    idx0=0
    ## code snippet from this https://www.kaggle.com/abhishek/text-extraction-using-bert-w-sentiment-inference
    for ind in (i for i, e in enumerate(x_str) if e == y_str[0]):
        if (x_str[ind: ind+len(y_str)] == y_str) or (x_str[ind: ind+len(y_str.replace(' ##',''))] == y_str.replace(' ##','')):
            idx0 = ind
            idx1 = ind + len(y_str) - 1
            break
    t = 0
    for offset,i in enumerate(x):
        if t +len(i)+1>idx0:
            break
        t = t+len(i)+1
    return offset

def create_targets(df, tokenizer):
    df['t_text'] = df['text'].apply(lambda x: tokenizer.tokenize(str(x)))
    df['t_selected_text'] = df['selected_text'].apply(lambda x: tokenizer.tokenize(str(x)))
    def func(row):
        x,y = row['t_text'],row['t_selected_text'][:]
        _offset = 0
        for offset in range(len(x)):
            _offset = offset
            d = dict(zip(x[offset:],y))
            #when k = v that means we found the offset
            check = [k==v for k,v in d.items()]
            if all(check)== True:
                break 
        targets = [0]*_offset + [1]*len(y) + [0]* (len(x)-_offset-len(y))
        ## should be same if not its a fail scenario because of  start or end issue 
        if len(targets) != len(x):
            offset = find_offset(x,y)
            targets = [0]*offset + [1]*len(y) + [0] * (len(x)-offset-len(y))
        return targets
    df['targets'] = df.apply(func,axis=1)
    return df

def _convert_to_transformer_inputs(text, tokenizer, max_sequence_length):
    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        return [input_ids, input_masks, input_segments]
    input_ids, input_masks, input_segments = return_id(text, None, 'longest_first', max_sequence_length)
    return [input_ids, input_masks, input_segments]

def compute_input_arrays(df, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df.iterrows()):
        t = str(instance.text)
        ids, masks, segments= _convert_to_transformer_inputs(t,tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns].values.tolist())

def create_model():
    id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    attn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = BertConfig() 
    config.output_hidden_states = True 
    bert_model = TFBertModel.from_pretrained('https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-tf_model.h5', config=config)
    _,_, hidden_states = bert_model(id, attention_mask=mask, token_type_ids=attn)
    h12 = tf.reshape(hidden_states[-1][:,0],(-1,1,768))
    h11 = tf.reshape(hidden_states[-2][:,0],(-1,1,768))
    h10 = tf.reshape(hidden_states[-3][:,0],(-1,1,768))
    h09 = tf.reshape(hidden_states[-4][:,0],(-1,1,768))
    concat_hidden = tf.keras.layers.Concatenate(axis=2)([h12, h11, h10, h09])
    x = tf.keras.layers.GlobalAveragePooling1D()(concat_hidden)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(MAX_TARGET_LEN, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[id, mask, attn], outputs=x)
    return model

def convert_pred_to_text(df,pred):
    temp_output = []
    for idx,p in enumerate(pred):
        indexes = np.where(p>0.5)
        current_text = df['t_text'][idx]
        if len(indexes[0])>0:
            start = indexes[0][0]
            end = indexes[0][-1]
        else:
            start = 0
            end = len(current_text)
        ### < was written previously but it should be > 
        ### (means model goes into padding tokens then restrict till the end of text)
        ### Thanks Davide Romano for pointing this out
        if end >= len(current_text):
            end = len(current_text)
        temp_output.append(' '.join(current_text[start:end+1]))
    return temp_output

def correct_op(row):
    placeholder = row['temp_output']
    for original_token in str(row['text']).split():
        token_str = ' '.join(tokenizer.tokenize(original_token))
        placeholder = placeholder.replace(token_str,original_token,1)
    return placeholder

def replacer(row):
    if row['sentiment'] == 'neutral':
        return row['text']
    else:
        return row['temp_output2']

train_df = read_train()
test_df = read_test()
submission_df = read_submission()

train_df = clean_df(train_df)
test_df = clean_df(test_df, train=False)

tokenizer = BertTokenizer.from_pretrained('https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt')
MAX_TARGET_LEN = MAX_SEQUENCE_LENGTH = 108
train_df = create_targets(train_df, tokenizer)
test_df['t_text'] = test_df['text'].apply(lambda x: tokenizer.tokenize(str(x)))
train_df['targets'] = train_df['targets'].apply(lambda x :x + [0] * (MAX_TARGET_LEN-len(x)))
outputs = compute_output_arrays(train_df,'targets')
inputs = compute_input_arrays(train_df, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(test_df, tokenizer, MAX_SEQUENCE_LENGTH)

train_inputs = inputs
train_outputs = outputs

del inputs,outputs
K.clear_session()
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

if not os.path.exists('/kaggle/input/tweet-finetuned-bert-v2/finetuned_bert.h5'):
    # Training done in another private kernel
    model.fit(train_inputs, train_outputs, epochs=10, batch_size=32)
    model.save_weights(f'finetuned_bert.h5')
else:
    model.load_weights('/kaggle/input/tweet-finetuned-bert-v2/finetuned_bert.h5')

test_df = read_test()
test_df = clean_df(test_df, train=False)
test_df['t_text'] = test_df['text'].apply(lambda x: tokenizer.tokenize(str(x)))
test_inputs = compute_input_arrays(test_df, tokenizer, MAX_SEQUENCE_LENGTH) 
threshold = 0.5
predictions = model.predict(test_inputs, batch_size=32, verbose=1)
pred = np.where(predictions>threshold,1,0)
test_df['temp_output'] = convert_pred_to_text(test_df,pred)
gc.collect()
test_df['temp_output2'] = test_df.progress_apply(correct_op,axis=1)
submission_df['selected_text'] = test_df['temp_output2']
submission_df['selected_text'] = submission_df['selected_text'].str.replace(' ##','')
submission_df.to_csv('submission.csv',index=False)
submission_df.head(10)