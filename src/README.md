# Models for perdicting like_count_24h of Dcard forum posts
The ***main.py*** file contains a GRU model and a Multi-layer Perceptron(MLP) model that can be used to predict the like count of a post on Dcard 24 hours after it was published. Both the models take into account the like count and comment count in first 6 hours. Moreover, users can decide to standardize data or to take titles of the posts into account as well.

## Model Architecture
The MLP model used in this project comprises of three hidden layers, with the first layer consisting of 128 neurons, the second consisting of 512 neurons, and the third consisting of 128 neurons. The output layer is a dense layer with a single neuron. Each hidden layer in the model is followed by a ReLU activation layer. Additionally, a ReLU layer is included after the output layer, since the predicted values are always greater than 0.

The GRU model consists of an embedding layer, a GRU layer with 128 neurons, and a dense output layer with a single neuron.

Both of the models were compiled using Adam optimizer and the mean absolute percentage error(MAPE)

## Training

### Install Required Packages

To install the required packages
```console
pip install -r src\requirements.txt
```

### Obtain BERT Classifier and Regressor
Two ways to get bert classifier and bert regressor for the model:
* Train on your own
```console
python3 src\models\bert_classifier.py
python3 src\models\bert_regressor.py
```
* Download the trained models from the following link and place them under ***src\models\\***  
[bert_classifier](https://drive.google.com/file/d/17c_tBjXFINAqjwyBL19rCG79BkEIC6tO/view?usp=share_link)  
[bert_regressor](https://drive.google.com/file/d/10F2u_CoeGMZe8EyuaOXUjPtnFoj5cMuO/view?usp=share_link)

### Train the Model
To train the model, run the following command:
```console
python3 src\main.py -lr -batch_size -epochs -m -is_standardize -is_bert
```
Here are the descriptions of the command line arguments:
* ***-lr*** : The learning rate of the optimizer used for training. (defalut: 0.0001)
* ***-batch_size*** The size of a batch used to train the model in one iteration. (default: 32)
* ***-epochs***: The number of epochs to train the model. (default: 100)
* ***-m***: The model chosed to train and predict. Choose from ['mlp', 'I', 'II', 'III', 'IV', 'V']
* ***-is_standardize***: An optional parameter. If included, the data will be standardized before training
* ***-is_bert***: An optional parameter. If included, choose from ['classifier', 'regressor']. Can only be assigned if model equals to 'II'

## Evaluation
The performance of the model can be evaluated using MAPE. To calculate MAPE, the following formula can be used:
<img src="https://render.githubusercontent.com/render/math?math=MAPE = \frac{1}{n}*\sum_{t=0}^n(\lvert \frac{y-\hat{y}}{y}\rvert)">
Where ***n*** is the number of samples, $y$ is the true like count, and $\hat{y}$ is the prediceted like count.
## Result
After training the model, the predicted like count for each post will be outputted to a CSV file named ***'result.csv'***.

## Author
* Kai Hsin, Chen
