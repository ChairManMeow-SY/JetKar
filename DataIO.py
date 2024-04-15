import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image

image_size=224

rscd_classes = {
    0: 'dry_asphalt_severe',
    1: 'dry_asphalt_slight',
    2: 'dry_asphalt_smooth',
    3: 'dry_concrete_severe',
    4: 'dry_concrete_slight',
    5: 'dry_concrete_smooth',
    6: 'dry_gravel',
    7: 'dry_mud',
    8: 'fresh_snow',
    9: 'ice',
    10: 'melted_snow',
    11: 'water_asphalt_severe',
    12: 'water_asphalt_slight',
    13: 'water_asphalt_smooth',
    14: 'water_concrete_severe',
    15: 'water_concrete_slight',
    16: 'water_concrete_smooth',
    17: 'water_gravel',
    18: 'water_mud',
    19: 'wet_asphalt_severe',
    20: 'wet_asphalt_slight',
    21: 'wet_asphalt_smooth',
    22: 'wet_concrete_severe',
    23: 'wet_concrete_slight',
    24: 'wet_concrete_smooth',
    25: 'wet_gravel',
    26: 'wet_mud'
}

rscd_labels={
    'dry_asphalt_severe': 0,
    'dry_asphalt_slight': 1,
    'dry_asphalt_smooth': 2,
    'dry_concrete_severe': 3,
    'dry_concrete_slight': 4,
    'dry_concrete_smooth': 5,
    'dry_gravel': 6,
    'dry_mud': 7,
    'fresh_snow': 8,
    'ice': 9,
    'melted_snow': 10,
    'water_asphalt_severe': 11,
    'water_asphalt_slight': 12,
    'water_asphalt_smooth': 13,
    'water_concrete_severe': 14,
    'water_concrete_slight': 15,
    'water_concrete_smooth': 16,
    'water_gravel': 17,
    'water_mud': 18,
    'wet_asphalt_severe': 19,
    'wet_asphalt_slight': 20,
    'wet_asphalt_smooth': 21,
    'wet_concrete_severe': 22,
    'wet_concrete_slight': 23,
    'wet_concrete_smooth': 24,
    'wet_gravel': 25,
    'wet_mud': 26
}

def InitTrainDataset(root):
    all_train_imgs=[]
    all_labels=[]

    all_folders=os.listdir(root)
    for folder_name in all_folders:
        cur_label=rscd_labels[folder_name]
        cur_folder=os.path.join(root,folder_name)
        all_imgs=os.listdir(cur_folder)
        for img_name in all_imgs:
            all_train_imgs.append(os.path.join(cur_folder,img_name))
            all_labels.append(cur_label)
    
    return all_train_imgs,all_labels

def InitTestDataset(root):
    all_test_imgs=[]
    all_labels=[]
    all_imgs=os.listdir(root)
    for img_name in all_imgs:
        all_test_imgs.append(os.path.join(root,img_name))
        label_str=img_name.split('.')[0]
        label_str='_'.join(label_str.split('-')[1:])
        all_labels.append(rscd_labels[label_str])
    return all_test_imgs,all_labels

def SaveDataset(root,save_file):
    '''
    train_imgs,train_labels,
    valid_imgs,valid_labels,
    test_imgs,test_labels 
    '''
    all_train_imgs,all_labels=InitTrainDataset(os.path.join(root,'train'))
    all_valid_imgs,all_valid_labels=InitTestDataset(os.path.join(root,'vali_20k'))
    all_test_imgs,all_test_labels=InitTestDataset(os.path.join(root,'test_50k'))

    with open(save_file,'wb') as f:
        pickle.dump(all_train_imgs,f)
        pickle.dump(all_labels,f)
        pickle.dump(all_valid_imgs,f)
        pickle.dump(all_valid_labels,f)
        pickle.dump(all_test_imgs,f)
        pickle.dump(all_test_labels,f)

def ReadDataset(save_file):
    with open(save_file,'rb') as f:
        train_imgs=pickle.load(f)
        train_labels=pickle.load(f)
        valid_imgs=pickle.load(f)
        valid_labels=pickle.load(f)
        test_imgs=pickle.load(f)
        test_labels=pickle.load(f)
    return [train_imgs,train_labels],[valid_imgs,valid_labels],[test_imgs,test_labels]

def MakeDataset(in_data):
    all_imgs,all_labels=in_data
    transforms=NoStrategyTransform()
    return RSCDDataset(all_imgs,all_labels,transform=transforms)

def NoStrategyTransform():
    simple_transform=transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return simple_transform 

class RSCDDataset(Dataset):
    def __init__(self, img_paths,img_labels, train=True, transform=None):
        super(RSCDDataset, self).__init__()
        self.train = train
        self.transform = transform
        self.all_imgs=img_paths
        self.all_labels=img_labels
    
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index):
        img_path=self.all_imgs[index]
        img_label=self.all_labels[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img,img_label
    
    def GetAllImageName(self):
        return self.all_imgs
    
    def GetAllLabels(self):
        return self.all_labels

def MakeAllDataloader(train_dataset,valid_dataset,test_dataset,batch_size,num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader,valid_loader,test_loader
