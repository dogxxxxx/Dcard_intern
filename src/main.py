import argparse
import logging
from models.dataset import MyDataset
from models.gru import GRUModel
from models.mlp import MLP
import utils.loggings as loggings
import numpy as np
import pandas as pd
import utils.preprocessing as preprocessing
from torch.utils.data import DataLoader
import utils.train as train

def main():
    config = {
        'train_path': 'data\\intern_homework_train_dataset.csv',
        'val_path': 'data\\intern_homework_public_test_dataset.csv',
        'test_path': 'data\\intern_homework_private_test_dataset.csv',
        'mlp_hidden_dims': [128,512, 128],
        'mlp_drop': ['title', 'created_at', 'forum_id', 'author_id', 'forum_stats'],
        'numeric_cols': ['like_count_1h', 'like_count_2h', 'like_count_3h', 'like_count_4h', 'like_count_5h', 'like_count_6h'
                         'comment_count_1h', 'comment_count_2h', 'comment_count_3h', 'comment_count_4h', 'comment_count_5h', 'comment_count_6h'],
        'gru_hidden_size': 128,
        'result_save_path': '.\\result.csv',
    }

    logging.info('-----Load data-----')

    if is_standardization:
        x_scaler, y_scaler, x_train, x_val, y_train, y_val, x_test = preprocessing.standardize(config=config)
    else:
        x_train, x_val, y_train, y_val, x_test = preprocessing.normal(config=config)

    if chosen_model=='gru':
        # reshape the input data to fit gru model
        x_train = x_train.reshape(-1,6,2)
        x_val = x_val.reshape(-1,6,2)
        x_test = x_test.reshape(-1,6,2)

    if chosen_model=='II':
        title_train, title_val, title_test = preprocessing.get_title_emb(config=config, model=is_bert)
        x_train = np.c_[x_train, title_train]
        x_val = np.c_[x_val, title_val]
        x_test = np.c_[x_test, title_test]

    if chosen_model=='III':
        x_train, x_val, x_test = preprocessing.time_cat(config=config, x_train=x_train, x_val=x_val, x_test=x_test)

    if chosen_model=='IV':
        x_train, x_val, x_test = preprocessing.get_mean(config=config, x_train=x_train, x_val=x_val, x_test=x_test, chosen_model=chosen_model)
        x_train, x_val, x_test = preprocessing.time_cat(config=config, x_train=x_train, x_val=x_val, x_test=x_test)

    if chosen_model=='V':
        x_train, x_val, x_test = preprocessing.get_mean(config=config, x_train=x_train, x_val=x_val, x_test=x_test, chosen_model=chosen_model)
        x_train, x_val, x_test = preprocessing.time_cat(config=config, x_train=x_train, x_val=x_val, x_test=x_test)
    
    train_dataset = MyDataset(x_train, y_train)
    val_dataset = MyDataset(x_val, y_val)
    test_dataset = MyDataset(x_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)

    if chosen_model!='gru':
        logging.info('-----train MLP-----')
        input_size = x_train.shape[1]
        model = MLP(input_size=input_size, hidden_size=config['mlp_hidden_dims'])
        mlp_best_model_dict = train.train_mlp(model=model, train_loader=train_loader, val_loader=val_loader, lr=lr, epochs=epochs)
        mlp_best_model  = MLP(input_size=input_size, hidden_size=config['mlp_hidden_dims'])
        mlp_best_model.load_state_dict(mlp_best_model_dict)
        mlp_predictions = train.test_mlp(model=mlp_best_model, test_loader=test_loader)
        if is_standardization:
            mlp_predictions = np.round(y_scaler.inverse_transform(mlp_predictions))
        else:
            mlp_predictions = np.round(mlp_predictions)
        pd.DataFrame(mlp_predictions, columns=['like_count_24h']).to_csv(config['result_save_path'], header=True, index=False)

    if chosen_model=='gru':
        logging.info('-----train GRU-----')
        input_size = x_train.shape[2]
        model = GRUModel(input_size, hidden_size=config['gru_hidden_size'])
        gru_best_model_dict = train.train_gru(model=model, train_loader=train_loader, val_loader=val_loader, lr=lr, epochs=epochs)
        gru_best_model = GRUModel(input_size, hidden_size=config['gru_hidden_size'])
        gru_best_model.load_state_dict(gru_best_model_dict)
        gru_predictions = train.test_gru(model=gru_best_model, test_loader=test_loader)
        if is_standardization:
            gru_predictions = np.round(y_scaler.inverse_transform(gru_predictions))
        else:
            gru_predictions = np.round(gru_predictions)
        pd.DataFrame(gru_predictions, columns=['like_count_24h']).to_csv(config['result_save_path'], header=True, index=False)
    

if __name__ == '__main__':
    loggings.set_logging('src\\logging.txt')
    logger = logging.getLogger()

    logger.info('-----Set up Parameters-----')
    parser = argparse.ArgumentParser(description='set up parameters')

    parser.add_argument("-lr", help="Set learning rate", nargs='?', type=float, const=0.0001, default=0.0001)
    parser.add_argument("-batch_size", help="Set batch size", nargs='?', type=int, const=32, default=32)
    parser.add_argument("-epochs", help="Set number of epochs", nargs='?', type=int, const=100, default=100)
    parser.add_argument("-m", help="Select model to use", required=True, choices=['gru', 'I', 'II', 'III', 'IV', 'V'])
    parser.add_argument("-is_standardization", help="Normalize data or not", action='store_true')
    if parser.parse_known_args()[0].m=='II':
        parser.add_argument("-is_bert", help="Use Bert classifier or regressor to get information from title or not using bert", choices=['classifier', 'regressor'], required=True)
    args = parser.parse_args()

    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    chosen_model = args.m
    if chosen_model=='II':
        is_bert = args.is_bert
    is_standardization = args.is_standardization
    logger.info('training start')
    main()
    logger.info('training end')
