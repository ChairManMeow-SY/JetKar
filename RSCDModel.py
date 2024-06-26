import torchvision.models as models
import torch
import torch.nn as nn
import os
from DataIO import NoStrategyTransform

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_classes=27
batch_size=128
image_size=224
training_epochs=5
num_workers=8

def Generate

class RoadSurfaceModel():
    def __init__(self,in_criterian=None,in_lr=0.001,use_) -> None:
        self.efficient_model=None
        self.data_transformer=None

        self.train_dataset=None
        self.valid_dataset=None
        self.test_dataset=None

        self.train_loader=None
        self.valid_loader=None
        self.test_loader=None

        self.save_folder=None
        self.save_log_file=None

        # training components
        if in_criterian is None:
            self.criterian=nn.CrossEntropyLoss()
        else:
            self.criterian=in_criterian

        self.lr=in_lr

    
    def init_save_folder(self,save_folder=None):
        if save_folder is None:
            self.save_folder='./save_model'
        else:
            self.save_folder=save_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    def initall(self,b_training=False,model_path=None,save_folder=None):
        self.init_save_folder(save_folder)
        if b_training:
            self.def_model(True)
            for param in self.efficient_model.parameters():
                param.requires_grad = True
        else:
            self.def_model(False)
            self.load_parameters(model_path)
            for param in self.efficient_model.parameters():
                param.requires_grad = False
        self.efficient_model.to(device)

    def rebuild_model(self,model_path=None):
        self.efficient_model = None
        self.def_model()
        self.load_parameters(model_path)

    def def_model(self,b_pretrained=False):
        # for the first time
        self.efficient_model = models.efficientnet_b0(pretrained=b_pretrained)
        in_f_num=self.efficient_model.classifier[1].in_features
        '''
        self.efficient_model.classifier[1]=nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_f_num, num_classes),
            nn.Softmax
        )
        '''
        self.efficient_model.classifier[1]=torch.nn.Linear(in_f_num,num_classes)
    
    def load_parameters(self,model_path):
        if model_path is None:
            return
        else:
            self.efficient_model.load_state_dict(torch.load(model_path))

    def save_model_parameters(self,save_path):
        torch.save(self.efficient_model.state_dict(), save_path)
    
    def train(self,epoch_num,log_file=None):
        self.save_log_file=log_file
        #criterian = torch.nn.CrossEntropyLoss()
        criterian =self.criterian
        optimizer = torch.optim.Adam(self.efficient_model.parameters(), lr=self.lr)

        self.efficient_model.to(device)
        self.efficient_model.train()

        all_training_loss=[]
        all_training_acc=[]

        all_valid_loss=[]
        all_valid_acc=[]

        self.log_save('[INFO] start training...\n')

        for epoch in range(epoch_num):
            trainning_loss,trainning_acc=self.train_one_epoch(optimizer,criterian)
            valid_loss,valid_acc=self.validate(criterian)

            all_training_loss.append(trainning_loss)
            all_training_acc.append(trainning_acc)

            all_valid_acc.append(valid_acc)
            all_valid_loss.append(valid_loss)

            if log_file is None:
                print*(f'[INFO] epoch_{epoch+1} starts......\n')
                print(f'[INFO] epoch_{epoch+1},trainning_loss:{trainning_loss},trainning_acc:{trainning_acc}\n')
                print(f'[INFO] epoch_{epoch+1},valid_loss:{valid_loss},valid_acc:{valid_acc}\n')
            else:
                with open(log_file,'a') as fid:
                    fid.write(f'[INFO] epoch_{epoch+1}\n')
                    fid.write(f'[INFO] epoch_{epoch+1},trainning_loss:{trainning_loss},trainning_acc:{trainning_acc}\n')
                    fid.write(f'[INFO] epoch_{epoch+1},valid_loss:{valid_loss},valid_acc:{valid_acc}\n')

            save_path=os.path.join(self.save_folder,f'epoch_{epoch+1}.pth')
            self.save_model_parameters(save_path)
        
        print('[INFO] finish training')

    def train_one_epoch(self,optimizer,creterian):
        self.efficient_model.train()
        training_loss=0
        training_accuracy=0

        counter=0
        for data, label in self.train_loader:
            print(f'[INFO] Now run batch {counter}')
            counter+=1
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = self.efficient_model(data)
            loss = creterian(output, label)
            loss.backward()
            optimizer.step()
            training_loss+=loss.item()
            training_accuracy+=torch.sum(torch.argmax(output, 1) == label).item()
        training_loss /= len(self.train_loader)
        training_accuracy /= len(self.train_loader)
        return training_loss,training_accuracy 
    
    def validate(self,creterian):
        self.efficient_model.eval()
        valid_loss=0
        valid_accuracy=0
        with torch.no_grad():
            for data,label in self.valid_loader:
                data, label = data.to(device), label.to(device)
                output = self.efficient_model(data)
                loss = creterian(output, label)
                valid_loss+=loss.item()
                valid_accuracy+=torch.sum(torch.argmax(output, 1) == label).item()
        valid_loss /= len(self.valid_loader)
        valid_accuracy /= len(self.valid_loader)
        return valid_loss,valid_accuracy

    def eval(self,log_file):
        self.save_log_file=log_file
        self.efficient_model.eval()
        test_accuracy=0
        with torch.no_grad():
            for data,label in self.test_loader:
                data, label = data.to(device), label.to(device)
                output = self.efficient_model(data)
                test_accuracy+=torch.sum(torch.argmax(output, 1) == label).item()
        test_accuracy /= (len(self.test_loader)*batch_size)
        self.log_save("The test acc is %f" % test_accuracy)
        return test_accuracy
    
    def SetTransformer(self,in_transform):
        self.data_transformer=in_transform
    
    def eval_jetour(self,jet_loader,label_to_jetour,log_file):
        self.save_log_file=log_file
        self.efficient_model.eval()
        test_accuracy=0
        test_count=0
        all_output_properties=[]

        all_origin_labels=[]
        all_out_labels=[]

        with torch.no_grad():
            for data,label in jet_loader:
                data, label = data.to(device), label.to(device)
                output = self.efficient_model(data)
                all_output_properties.append(output)
                origin_label=torch.argmax(output, 1)
                all_origin_labels.append(origin_label)
                out_label=origin_label.clone()
                for i in range(origin_label.shape[0]):
                    tmp=int(out_label[i])
                    if tmp in label_to_jetour.keys():
                        out_label[i]=label_to_jetour[int(out_label[i])]
                test_count+=len(label)
                all_out_labels.append(out_label)
                test_accuracy+=torch.sum(out_label == label).item()
        test_accuracy /= test_count
        self.log_save("The test acc is %f" % test_accuracy)
        print("The test acc is %f" % test_accuracy)
        return test_accuracy,all_origin_labels,all_out_labels,all_output_properties

    def eval_single_image(self,in_img):
        if self.data_transformer is None:
            self.SetTransformer(NoStrategyTransform())
        in_img=self.data_transformer(in_img)
        self.efficient_model.eval()
        with torch.no_grad():
            in_img=in_img.to(device)
            output = self.efficient_model(in_img)
            ret_label=torch.argmax(output, 1) 
        return ret_label

    def log_save(self,message_str):
        if self.save_log_file is None:
            print(message_str)
        else:
            with open(self.save_log_file,'a') as f:
                f.write(message_str)
    