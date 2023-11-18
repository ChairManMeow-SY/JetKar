import os
import RSCDModel
import DataIO
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

## init dataset ##
dataset_root='D:/Work/RSCD/RSCD_dataset'
save_dataset_file=dataset_root+'/all_dataset.pkl'
DataIO.SaveDataset(dataset_root,save_dataset_file)

## init model ##
batch_size=RSCDModel.batch_size
num_workers=RSCDModel.num_workers
model = RSCDModel.RoadSurfaceModel()
train_data,valid_data,test_data=DataIO.ReadDataset(save_dataset_file)
train_loader,valid_loader,test_loader=DataIO.MakeAllDataloader(train_data,valid_data,test_data,batch_size,num_workers)
model.initall()

