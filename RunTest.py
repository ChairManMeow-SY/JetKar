import os
import RSCDModel
import DataIO
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset_root='/data/IDTZSY/RSCD/dataset'
save_dataset_file='/data/IDTZSY/RSCD/dataset/all_dataset.pkl'

batch_size=RSCDModel.batch_size
num_workers=RSCDModel.num_workers
rscd_model=RSCDModel.RoadSurfaceModel()

train_data,valid_data,test_data=DataIO.ReadDataset(save_dataset_file)
train_dataset=DataIO.MakeDataset(train_data)
valid_dataset=DataIO.MakeDataset(valid_data)
test_dataset=DataIO.MakeDataset(test_data)

train_loader,valid_loader,test_loader=DataIO.MakeAllDataloader(train_dataset,valid_dataset,test_dataset,batch_size,num_workers)

rscd_model.initall(False,'/code/epoch_8.pth','/output/save_model')

rscd_model.train_dataset=train_data
rscd_model.train_loader=train_loader

rscd_model.test_dataset=test_data
rscd_model.test_loader=test_loader

rscd_model.valid_dataset=valid_data
rscd_model.valid_loader=valid_loader

rscd_model.eval("/output/log.txt")