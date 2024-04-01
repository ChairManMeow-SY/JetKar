import DataIO
import os
import json
from torch.utils.data import DataLoader
import RSCDModel

jetour_root='/data/IDTZSY/RSCD/Jetour/dataset/'
front_view_images='front_view_images'
label_file='final_label.json'
bev_images='bev_images'
source_images='src_images'

def GetJetourDataset(jetour_path,json_file):
    all_imgs=[]
    all_labels=[]
    with open(json_file,'r') as fid:
        all_json_data=json.load(fid)
    
    for json_data in all_json_data:
        img_name=json_data['image_name']
        img_path=os.path.join(jetour_path,img_name)

        label=json_data['image_attr']["label_1"]
        label=int(label)
        all_imgs.append(img_path)
        all_labels.append(label)
    
    jetour_dataset=DataIO.RSCDDataset(all_imgs,all_labels,False,transform=DataIO.NoStrategyTransform())

    return jetour_dataset

def GetJetourDataLoader(jetour_path,json_file,batch_size=RSCDModel.batch_size,shuffle=False,num_workers=RSCDModel.num_workers):
    jetour_dataset=GetJetourDataset(jetour_path,json_file)
    jetour_loader=DataLoader(jetour_dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return jetour_loader

def GetJetourasphaltDataset_SpecificLabel(jetour_path,json_file,in_label=1):
    '''
    1: 水泥
    2：雪地 
    '''
    all_imgs=[]
    all_labels=[]
    in_label=int(in_label)

    with open(json_file,'r') as fid:
        all_json_data=json.load(fid)
    
    for json_data in all_json_data:
        img_name=json_data['image_name']
        img_path=os.path.join(jetour_path,img_name)

        label=json_data['image_attr']["label_1"]
        if label==in_label:
            all_imgs.append(img_path)
            all_labels.append(label)
    
    jetour_dataset=DataIO.RSCDDataset(all_imgs,all_labels,False,transform=DataIO.NoStrategyTransform())

    return jetour_dataset

def MapRSCDLabelToJetour():
    '''
    1: asphalt, concrete
    2: snow + ice 
    '''
    jetour_labels={}
    jetour_labels[1]=[]
    jetour_labels[2]=[]
    jetour_classes={}

    rscd_labels=DataIO.rscd_labels
    for key,item in rscd_labels.items():
        if 'asphalt' in key: #or 'concrete' in key:
            jetour_labels[1].append(item)
        elif 'snow' in key or 'ice' in key:
            jetour_labels[2].append(item)
    
    for id in jetour_labels[1]:
        jetour_classes[id]=1
    for id in jetour_labels[2]:
        jetour_classes[id]=2
    return jetour_classes,jetour_labels 