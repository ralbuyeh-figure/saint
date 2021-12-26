import torch
from torch import nn
from models import SAINT

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np

class FakeParser:
    pass
  
# Driver's code
opt = FakeParser()

opt.dset_id = 1461
opt.task = "binary"
opt.attentiontype = "colrow"
opt.cont_embeddings = "MLP"
opt.vision_dset = False
opt.embedding_size = 32
opt.transformer_depth = 6
opt.attention_heads = 8
opt.attention_dropout = 0.1
opt.ff_dropout = 0.1
opt.attentiontype = "colrow"
opt.optimizer = "AdamW"
#parser = argparse.ArgumentParser()

#parser.add_argument('--dset_id', required=True, type=int)
#parser.add_argument('--vision_dset', action = 'store_true')
#parser.add_argument('--task', required=True, type=str,choices = ['binary','multiclass','regression'])
#parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
#parser.add_argument('--embedding_size', default=32, type=int)
#parser.add_argument('--transformer_depth', default=6, type=int)
#parser.add_argument('--attention_heads', default=8, type=int)
#parser.add_argument('--attention_dropout', default=0.1, type=float)
#parser.add_argument('--ff_dropout', default=0.1, type=float)
#parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

#parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
opt.scheduler = "cosine"
#parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

opt.lr = 0.0001
#parser.add_argument('--lr', default=0.0001, type=float)

opt.epochs = 100
#parser.add_argument('--epochs', default=100, type=int)

opt.batchsize = 256
#parser.add_argument('--batchsize', default=256, type=int)

opt.savemodelroot = "./bestmodels"
#parser.add_argument('--savemodelroot', default='./bestmodels', type=str)

opt.run_name = "testrun"
#parser.add_argument('--run_name', default='testrun', type=str)

opt.set_seed = 1
#parser.add_argument('--set_seed', default= 1 , type=int)

opt.dset_seed = 5
#parser.add_argument('--dset_seed', default= 5 , type=int)

opt.active_log = True
#parser.add_argument('--active_log', action = 'store_true')

opt.pretrain = True
#parser.add_argument('--pretrain', action = 'store_true')

opt.pretrain_epochs = 50
#parser.add_argument('--pretrain_epochs', default=50, type=int)

opt.pt_tasks = ["contrastive", "denoising"]

#parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])

opt.pt_aug = []

#parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])

opt.pt_aug_lam = 0.1
#parser.add_argument('--pt_aug_lam', default=0.1, type=float)

opt.mixup_lam = 0.3
#parser.add_argument('--mixup_lam', default=0.3, type=float)

opt.train_mask_prob = 0
#parser.add_argument('--train_mask_prob', default=0, type=float)

opt.mask_prob = 0
#parser.add_argument('--mask_prob', default=0, type=float)

opt.ssl_avail_y = 0
#parser.add_argument('--ssl_avail_y', default= 0, type=int)

opt.pt_projhead_style = "diff"

#parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])

opt.nce_temp = 0.7

#parser.add_argument('--nce_temp', default=0.7, type=float)

opt.lam0 = 0.5
opt.lam1 = 10
opt.lam2 = 1
opt.lam3 = 10

opt.final_mlp_style = "sep"

#parser.add_argument('--lam0', default=0.5, type=float)
#parser.add_argument('--lam1', default=10, type=float)
#parser.add_argument('--lam2', default=1, type=float)
#parser.add_argument('--lam3', default=10, type=float)
#parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])


#opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb
    if opt.pretrain:
        wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:
        if opt.task=='multiclass':
            wandb.init(project="saint_v2_all_kamal", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
        else:
            wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
   

#### gutting the data preprocessing

import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset

ds_id = opt.dset_id
seed = opt.dset_seed
task = opt.task

datasplit=[.65, .15, .2]

np.random.seed(seed)
dataset = openml.datasets.get_dataset(ds_id)

X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                target=dataset.default_target_attribute)

# x is a pandas dataframe with a bunch of features, mixed continuous and float
# y is a pandas series with distinct caegories, '1' and '2'
# categorical indicator is a list of booleans where the value corresponds to the column index
# attribute names is like above but with names in place of boolean

if ds_id == 42178:
    categorical_indicator = [True, False, True, True, False, True, True, True, True, True, True, True, True, True, True,
                             True, True, False, False]
    tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
    X['TotalCharges'] = [float(i) for i in tmp]
    y = y[X.TotalCharges != 0]
    X = X[X.TotalCharges != 0]
    X.reset_index(drop=True, inplace=True)
    print(y.shape, X.shape)
if ds_id in [42728, 42705, 42729, 42571]:
    # import ipdb; ipdb.set_trace()
    X, y = X[:50000], y[:50000]
    X.reset_index(drop=True, inplace=True)
categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
# this just specifies the categorical column names
cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
# this is the continuous columns, the disjoint of feature names where you remove the categorical columsn or whatever

cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
# indexes of categorical columns.. for some reason
con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
# indexes of continuous columns
for col in categorical_columns:
    X[col] = X[col].astype("object")
# converting all the categoricals to type object

X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))
# apply train test val flag

train_indices = X[X.Set == "train"].index
valid_indices = X[X.Set == "valid"].index
test_indices = X[X.Set == "test"].index
# int64 index of the corresponding indices to flag

X = X.drop(columns=['Set'])
# drop that flag column
temp = X.fillna("MissingValue")
# fillna as other...
nan_mask = temp.ne("MissingValue").astype(int)
# returns a dataframe of 1s where value is not missing value... so mostly a matrix of 1s.

cat_dims = []
for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
    X[col] = X[col].fillna("MissingValue")
    l_enc = LabelEncoder()
    X[col] = l_enc.fit_transform(X[col].values)
    cat_dims.append(len(l_enc.classes_))
# apply arbitrary integer values to categorical columns...
# cat dims is the number of distinct categories for each categorical column
# watch out here, they're not really being mindful of leakage.

for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
    X.fillna(X.loc[train_indices, col].mean(), inplace=True)
# mean impute the continuous columns... that's bad because i don't see them doing any of that using the training params
# on the val and test, we'll see


y = y.values
if task != 'regression':
    l_enc = LabelEncoder()
    y = l_enc.fit_transform(y)
# label encoding the y vector to be 0s and 1s..

def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'

    y_d = {
        'data': y[indices].reshape(-1, 1)
    }
    return x_d, y_d

# above function returns x_d which is a dictionary of numpy array of x data values, and then the mask, and then
# row filtered based on an index

X_train, y_train = data_split(X, y, nan_mask, train_indices)
X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
X_test, y_test = data_split(X, y, nan_mask, test_indices)

train_mean, train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0), np.array(
    X_train['data'][:, con_idxs], dtype=np.float32).std(0)
train_std = np.where(train_std < 1e-6, 1e-6, train_std)
# import ipdb; ipdb.set_trace()


####DONE

print('Downloading and processing the dataset, it might take some time.')
#cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(opt.dset_id, opt.dset_seed,opt.task, datasplit=[.65, .15, .2])
# I scrubbed the above line because I gutted it in the above section

continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)

##### Setting some hyperparams based on inputs and dataset
_,nfeat = X_train['data'].shape
# this just gets the number of features in the x matrix

if nfeat > 100:
    opt.embedding_size = min(8,opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = min(4,opt.attention_heads)
    opt.attention_dropout = 0.8
    opt.embedding_size = min(32,opt.embedding_size)
    opt.ff_dropout = 0.8

print(nfeat,opt.batchsize)
print(opt)

if opt.active_log:
    wandb.config.update(opt)

##### gutting the datasetcatcon class


# # class DataSetCatCon(Dataset):
# #     def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):
#
# _X = X_train
# # that data dict thing
# _Y = y_train
# # the y dict
# _cat_cols = cat_idxs
# # indices of categorical columns
# _task = opt.dtask
# # 'clf' in our case
# _continuous_mean_std = continuous_mean_std
#
# _cat_cols = list(_cat_cols)
# # redundant
# _X_mask = _X['mask'].copy()
# # getting the mask of that data dict
# _X = _X['data'].copy()
# # getting the X element of that data dict
# _con_cols = list(set(np.arange(_X.shape[1])) - set(_cat_cols))
# # the continuous column indices
# _X1 = _X[:, _cat_cols].copy().astype(np.int64)  # categorical columns
# # broken off categorical columns
# _X2 = _X[:, _con_cols].copy().astype(np.float32)  # numerical columns
# # broken off numerical columns
#
# _X1_mask = _X_mask[:, _cat_cols].copy().astype(np.int64)  # categorical columns
# # broken off categorical missing value mask
#
# _X2_mask = _X_mask[:, _con_cols].copy().astype(np.int64)  # numerical columns
# # broken off numerical missing value mask
#
# if task == 'clf':
#     _y = _Y['data']  # .astype(np.float32)
# else:
#     _y = _Y['data'].astype(np.float32)
# # just grabbing that y vector
#
# _cls = np.zeros_like(_y, dtype=int)
# # get a bunch of zeros in the same dimensionality as y vector
#
# _cls_mask = np.ones_like(_y, dtype=int)
# # get a bunch of ones in same dimensionality as y vector
#
#
# if continuous_mean_std is not None:
#     _mean, _std = continuous_mean_std
#     _X2 = (_X2 - _mean) / _std

# z normalize only the continuous x columns

    #
    # # def __len__(self):
    #     return len(self.y)
    #
    # # def __getitem__(self, idx):
    #     # X1 has categorical data, X2 has continuous
    #     return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], np.concatenate(
    #         (self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

# note that they are not converting it to a torch tensor.. they must do it at some point...


train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask,continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

# a single element looks like this:

# (array([0, 4, 1, 2, 0, 1, 0, 2, 8, 3]),
# this one is like the categorical vector, with a leading zero appended to it. Andrew suggests it might be the token they mention in the paper
#  array([ 1.5994084 ,  0.25852484, -1.2972844 ,  0.01338016, -0.5672942 ,
#         -0.41616383, -0.2361183 ], dtype=float32),
# this one is the continuous element of the vector
#  array([0]),
#  this is the y-element of a vector
#  array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
#  │broken off categorical missing value mask with a leading 1 appended to it. that's because that 0 element is not missing
#  array([1, 1, 1, 1, 1, 1, 1]))
# broken off numerical missing value mask

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs,opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.


# note in this implementation, we are passing a bunch of hyperparams directly to the model class

model = SAINT(
categories = tuple(cat_dims), # remember there's a leading 1 here.. array([ 1, 12,  3,  4,  2,  2,  2,  3, 12,  4])
num_continuous = len(con_idxs), # continuous indices: [0, 5, 9, 11, 12, 13, 14]
dim = opt.embedding_size, # via config: 32
dim_out = 1, # I think just a single output?
depth = opt.transformer_depth,  # in our case, overridden to 1
heads = opt.attention_heads,   # in our case, 4
attn_dropout = opt.attention_dropout,  # in our case, overridden to 0.8
ff_dropout = opt.ff_dropout,  # in our case, overridden to 0.8
mlp_hidden_mults = (4, 2),  #  I think these are feedforward hidden layers
cont_embeddings = opt.cont_embeddings, # i forget what this does
attentiontype = opt.attentiontype, # colrow aka vanilla SAINT
final_mlp_style = opt.final_mlp_style, # it's 'sep' but honestly i don't know what this does
y_dim = y_dim #dimensinoality of objective, in this case 2
)
vision_dset = opt.vision_dset

if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise'case not written yet'

model.to(device)


if opt.pretrain:
    from pretraining import SAINT_pretrain
    model = SAINT_pretrain(model, cat_idxs,X_train,y_train, continuous_mean_std, opt,device)

## Choosing the optimizer

if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
print('Training begins now.')
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)           
        reps = model.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        
        y_outs = model.mlpfory(y_reps)
        if opt.task == 'regression':
            loss = criterion(y_outs,y_gts) 
        else:
            loss = criterion(y_outs,y_gts.squeeze()) 
        loss.backward()
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    # print(running_loss)
    if opt.active_log:
        wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss, 
        'loss': loss.item()
        })
    if epoch%5==0:
            model.eval()
            with torch.no_grad():
                if opt.task in ['binary','multiclass']:
                    accuracy, auroc = classification_scores(model, validloader, device, opt.task,vision_dset)
                    test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task,vision_dset)

                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                        (epoch + 1, accuracy,auroc ))
                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                        (epoch + 1, test_accuracy,test_auroc ))
                    if opt.active_log:
                        wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })     
                        wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })  
                    if opt.task =='multiclass':
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                    else:
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                        # if auroc > best_valid_auroc:
                        #     best_valid_auroc = auroc
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy               
                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

                else:
                    valid_rmse = mean_sq_error(model, validloader, device,vision_dset)    
                    test_rmse = mean_sq_error(model, testloader, device,vision_dset)  
                    print('[EPOCH %d] VALID RMSE: %.3f' %
                        (epoch + 1, valid_rmse ))
                    print('[EPOCH %d] TEST RMSE: %.3f' %
                        (epoch + 1, test_rmse ))
                    if opt.active_log:
                        wandb.log({'valid_rmse': valid_rmse ,'test_rmse': test_rmse })     
                    if valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        best_test_rmse = test_rmse
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            model.train()
                


total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.task =='binary':
    print('AUROC on best model:  %.3f' %(best_test_auroc))
elif opt.task =='multiclass':
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
else:
    print('RMSE on best model:  %.3f' %(best_test_rmse))

if opt.active_log:
    if opt.task == 'regression':
        wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep':best_test_rmse , 
        'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })        
    else:
        wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc , 
        'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })
