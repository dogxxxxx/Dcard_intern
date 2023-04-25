from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from models.bert_dataset_model import BertClassifier, BertDataset, BertRegressor
from torch.utils.data import DataLoader
from tqdm import tqdm

def minmax(config):
    cols = ['like_count_1h', 'comment_count_1h',
            'like_count_2h', 'comment_count_2h',
            'like_count_3h', 'comment_count_3h',
            'like_count_4h', 'comment_count_4h',
            'like_count_5h', 'comment_count_5h',
            'like_count_6h', 'comment_count_6h']
    train = pd.read_csv(config['train_path']).drop(config['mlp_drop'], axis=1)
    val = pd.read_csv(config['val_path']).drop(config['mlp_drop'], axis=1)
    test = pd.read_csv(config['test_path']).drop(config['mlp_drop'], axis=1).values
    x_train = train.drop('like_count_24h', axis=1).values
    x_val = val.drop('like_count_24h', axis=1).values
    y_train = train['like_count_24h'].values.reshape(-1,1)
    y_val = val['like_count_24h'].values.reshape(-1,1)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)
    y_train = y_scaler.fit_transform(y_train)
    x_val = x_scaler.transform(x_val)
    y_val = y_scaler.transform(y_val)
    x_test = x_scaler.transform(test)
    return x_scaler, y_scaler, x_train, x_val, y_train, y_val, x_test

def standardize(config):
    # rearrange the order of dataframe to fit GRU model
    cols = ['like_count_1h', 'comment_count_1h',
            'like_count_2h', 'comment_count_2h',
            'like_count_3h', 'comment_count_3h',
            'like_count_4h', 'comment_count_4h',
            'like_count_5h', 'comment_count_5h',
            'like_count_6h', 'comment_count_6h']
    train = pd.read_csv(config['train_path']).drop(config['mlp_drop'], axis=1)
    val = pd.read_csv(config['val_path']).drop(config['mlp_drop'], axis=1)
    test = pd.read_csv(config['test_path']).drop(config['mlp_drop'], axis=1)[cols].values
    x_train = train.drop('like_count_24h', axis=1)[cols].values
    x_val = val.drop('like_count_24h', axis=1)[cols].values
    y_train = train['like_count_24h'].values.reshape(-1,1)
    y_val = val['like_count_24h'].values.reshape(-1,1)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    y_train = y_scaler.fit_transform(y_train)
    x_val = x_scaler.transform(x_val)
    y_val = y_scaler.transform(y_val)
    x_test = x_scaler.transform(test)
    return x_scaler, y_scaler, x_train, x_val, y_train, y_val, x_test

def normal(config):
    # rearrange the order of dataframe to fit GRU model
    cols = ['like_count_1h', 'comment_count_1h',
            'like_count_2h', 'comment_count_2h',
            'like_count_3h', 'comment_count_3h',
            'like_count_4h', 'comment_count_4h',
            'like_count_5h', 'comment_count_5h',
            'like_count_6h', 'comment_count_6h']
    train = pd.read_csv(config['train_path']).drop(config['mlp_drop'], axis=1)
    val = pd.read_csv(config['val_path']).drop(config['mlp_drop'], axis=1)
    test = pd.read_csv(config['test_path']).drop(config['mlp_drop'], axis=1)[cols].values
    x_train = train.drop('like_count_24h', axis=1)[cols].values
    x_val = val.drop('like_count_24h', axis=1)[cols].values
    y_train = train['like_count_24h'].values.reshape(-1,1)
    y_val = val['like_count_24h'].values.reshape(-1,1)
    return x_train, x_val, y_train, y_val, test

def get_title_emb(config, model):
    title_train = pd.read_csv(config['train_path'])['title']
    title_val = pd.read_csv(config['val_path'])['title']
    title_test = pd.read_csv(config['test_path'])['title']
    if model=='classifier':
        model = BertClassifier()
        state_dict = torch.load('src\\models\\best_bertclassifier.pt')
        model.load_state_dict(state_dict=state_dict)
    else:
        model = BertRegressor()
        state_dict = torch.load('src\\models\\best_bertregressor.pt')
        model.load_state_dict(state_dict=state_dict)

    train_set = BertDataset(df=title_train)
    val_set = BertDataset(df=title_val)
    test_set = BertDataset(df=title_test)

    train_loader = DataLoader(train_set, batch_size=144, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=144, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=144, shuffle=False)

    model.eval()
    with torch.no_grad():
        train_result=[]
        val_result=[]
        test_result=[]
        for batch in tqdm(train_loader):
            input_ids, attention_mask = batch
            output = model(input_ids, attention_mask)
            train_result = train_result + output.tolist()
        for batch in tqdm(val_loader):
            input_ids, attention_mask = batch
            output = model(input_ids, attention_mask)
            val_result = val_result + output.tolist()
        for batch in tqdm(test_loader):
            input_ids, attention_mask = batch
            output = model(input_ids, attention_mask)
            test_result = test_result + output.tolist()
    return np.array(train_result), np.array(val_result), np.array(test_result)


def time_cat(config, x_train, x_val, x_test):
    train = pd.read_csv(config['train_path'])['created_at'].values
    val = pd.read_csv(config['val_path'])['created_at'].values
    test = pd.read_csv(config['test_path'])['created_at'].values
    train_time = np.array([datetime.strptime(s, '%Y-%m-%d %H:%M:%S %Z').hour for s in train])
    val_time = np.array([datetime.strptime(s, '%Y-%m-%d %H:%M:%S %Z').hour for s in val])
    test_time = np.array([datetime.strptime(s, '%Y-%m-%d %H:%M:%S %Z').hour for s in test])
    train_bins = np.zeros((len(train_time), 4))
    val_bins = np.zeros((len(val_time), 4))
    test_bins = np.zeros((len(test_time), 4))  
    train_bins[np.where((train_time >= 0) & (train_time < 6))] = [1, 0, 0, 0]
    train_bins[np.where((train_time >= 6) & (train_time < 12))] = [0, 1, 0, 0]
    train_bins[np.where((train_time >= 12) & (train_time < 18))] = [0, 0, 1, 0]
    train_bins[np.where((train_time >= 18) & (train_time < 24))] = [0, 0, 0, 1]

    val_bins[np.where((val_time >= 0) & (val_time < 6))] = [1, 0, 0, 0]
    val_bins[np.where((val_time >= 6) & (val_time < 12))] = [0, 1, 0, 0]
    val_bins[np.where((val_time >= 12) & (val_time < 18))] = [0, 0, 1, 0]
    val_bins[np.where((val_time >= 18) & (val_time < 24))] = [0, 0, 0, 1]

    test_bins[np.where((test_time >= 0) & (test_time < 6))] = [1, 0, 0, 0]
    test_bins[np.where((test_time >= 6) & (test_time < 12))] = [0, 1, 0, 0]
    test_bins[np.where((test_time >= 12) & (test_time < 18))] = [0, 0, 1, 0]
    test_bins[np.where((test_time >= 18) & (test_time < 24))] = [0, 0, 0, 1]

    x_train = np.c_[x_train, train_bins]
    x_val = np.c_[x_val, val_bins]
    x_test = np.c_[x_test, test_bins]
    return x_train, x_val, x_test

def get_mean(config, x_train, x_val, x_test, chosen_model):
    train = pd.read_csv(config['train_path'])[['like_count_24h', 'forum_id', 'forum_stats', 'author_id']]
    val = pd.read_csv(config['val_path'])[['forum_id', 'forum_stats', 'author_id']]
    test = pd.read_csv(config['test_path'])[['forum_id', 'forum_stats', 'author_id']]
    fid_dict = train.groupby('forum_id').mean().to_dict()['like_count_24h']
    fst_dict = train.groupby('forum_stats').mean().to_dict()['like_count_24h']
    aid_dict = train.groupby('author_id').mean().to_dict()['like_count_24h']
    average = train['like_count_24h'].mean()

    train['mean_fid'] = train['forum_id'].apply(lambda x: fid_dict[x])
    train['mean_fst'] = train['forum_stats'].apply(lambda x: fst_dict[x])
    train['mean_aid'] = train['author_id'].apply(lambda x: aid_dict[x])
    x_train = np.c_[x_train, train.drop(['forum_id', 'forum_stats', 'author_id', 'like_count_24h'], axis=1)]

    val['mean_fid'] = val['forum_id'].apply(lambda x: fid_dict[x] if x in fid_dict else average)
    val['mean_fst'] = val['forum_stats'].apply(lambda x: fst_dict[x] if x in fst_dict else average)
    val['mean_aid'] = val['author_id'].apply(lambda x: aid_dict[x] if x in aid_dict else average)
    x_val = np.c_[x_val, val.drop(['forum_id', 'forum_stats', 'author_id'], axis=1)]

    test['mean_fid'] = test['forum_id'].apply(lambda x: fid_dict[x] if x in fid_dict else average)
    test['mean_fst'] = test['forum_stats'].apply(lambda x: fst_dict[x] if x in fst_dict else average)
    test['mean_aid'] = test['author_id'].apply(lambda x: aid_dict[x] if x in aid_dict else average)
    x_test = np.c_[x_test, test.drop(['forum_id', 'forum_stats', 'author_id'], axis=1)]

    if chosen_model=='V':
        x_train = np.delete(x_train, -1, 1)
        x_val = np.delete(x_val, -1, 1)
        x_test = np.delete(x_test, -1, 1)
    return x_train, x_val, x_test




















def fill_na(df, is_bert: bool):
    config={
        'drop_titles': ['created_at', 'forum_id', 'author_id', 'forum_stats', 'title'], ### may use forum_id or author_id later
        'drop_without_titles': ['created_at', 'forum_id', 'author_id', 'forum_stats']
    }
    if is_bert:
        df = df.drop(config['drop_without_titles'], axis=1) # drop the columns that will not be used
        df = df.dropna(subset=['title']) # if the title is missing, drop it.
        df = df.fillna(df.mean(numeric_only=True)) # fill missing values with the mean of the column
    else:
        df = df.drop(config['drop_titles'], axis=1) # drop the columns that will not be used
        df = df.fillna(df.mean(numeric_only=True)) # fill missing values with the mean of the column
    return df

def min_max():
    return

def split_feature_label(df, is_test=False):
    if is_test:
        feature = torch.tensor(df.values)
        return feature
    label = torch.tensor(df['like_count_24h'].values, dtype=torch.float32)
    feature = torch.tensor(df.iloc[:, 0:-1].values, dtype=torch.float32)
    return feature, label

def get_dataloader(x, y, batch_size, shuffle=False):
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader