import JetourDataset
import DataIO
import RSCDModel
import os
import json
import torch
import numpy as np


def GetLargetstClasses(in_prop,largest_num):
    '''
    return labels 
    '''
    prop_index=np.argsort(in_prop)
    ret_index=prop_index[-largest_num:]
    # not right
    norm_prop=np.e**in_prop
    norm_prop/=np.sum(norm_prop)

    return [(ret_index[i],norm_prop[ret_index[i]]) for i in range(len(ret_index))]

if __name__=='__main__':

    model_path='./epoch_8.pth'
    save_folder='/output/'
    log_file='/output/log.txt'

    jetour_root='/data/IDTZSY/RSCD/Jetour/dataset/'
    front_view_images='front_view_images'
    label_file='final_label.json'
    bev_images='bev_images'
    source_images='src_images'
    json_file='final_label.json'

    json_path=os.path.join(jetour_root,json_file)
    jetour_path=os.path.join(jetour_root,front_view_images)

    jetour_classes,jetour_labels=JetourDataset.MapRSCDLabelToJetour()


    jetour_model=RSCDModel.RoadSurfaceModel()
    jetour_model.initall(False,model_path,save_folder)


    jet_loader=JetourDataset.GetJetourDataLoader(jetour_path,json_path)

    tmp,all_origin_labels,all_out_labels,all_prop= jetour_model.eval_jetour(jet_loader,jetour_classes,log_file)

    all_origin_labels=torch.concatenate(all_origin_labels)
    all_origin_labels=all_origin_labels.to('cpu')
    all_origin_labels=all_origin_labels.numpy()

    all_prop=torch.concatenate(all_prop,0).to('cpu').numpy()

    all_out_labels=torch.concatenate(all_out_labels)
    all_out_labels=all_out_labels.to('cpu')
    all_out_labels=all_out_labels.numpy()

    all_jetour_labels=[]
    all_img_names=[]

    with open(json_path,'r') as fid:
        all_data=json.load(fid)
        for image_info in all_data:
            all_img_names.append(image_info['image_name'])
            all_jetour_labels.append(int(image_info['image_attr']['label_1']))

    asphalt_acc=0
    snow_acc=0
    asphalt_num=0
    snow_num=0

    all_bad_case={}
    all_bad_case[1]=[]
    all_bad_case[2]=[]
    best_prop_num=5

    for i in range(len(all_jetour_labels)):
        cur_prop=all_prop[i,:]

        best_prop_5=GetLargetstClasses(cur_prop,best_prop_num)

        if all_jetour_labels[i] ==1:
            asphalt_num+=1
            if all_out_labels[i] == all_jetour_labels[i]:
                asphalt_acc+=1
            else:
                all_bad_case[1].append([all_img_names[i],all_origin_labels[i],best_prop_5])

        if all_jetour_labels[i] ==2:
            snow_num+=1
            if all_out_labels[i] == all_jetour_labels[i]:
                snow_acc+=1
            else:
                all_bad_case[2].append([all_img_names[i],all_origin_labels[i],best_prop_5])

    asphalt_acc=asphalt_acc/asphalt_num
    snow_acc=snow_acc/snow_num

    rscd_classes=DataIO.rscd_classes

    with open('/output/bad_case.txt','w') as fid:
        fid.write('the asphalt bad cases:\n')
        for i in range(len(all_bad_case[1])):
            fid.write('img name: %s, output label is : %d (%s), ' % (all_bad_case[1][i][0],all_bad_case[1][i][1],rscd_classes[int(all_bad_case[1][i][1])]))
            cur_prop=all_bad_case[1][i][2]
            fid.write(f' the best {best_prop_num} prop is : ')
            for k in range(best_prop_num):
                fid.write(f'{cur_prop[k][0]}({rscd_classes[int(cur_prop[k][0])]}) :{cur_prop[k][1]} ')
            fid.write('\n')

        fid.write('#########################################\n')
        fid.write('#########################################\n')
        fid.write('#########################################\n')
        fid.write('the snow bad cases:\n')
        for i in range(len(all_bad_case[2])):
            fid.write('img name: %s, output label is : %d (%s), ' % (all_bad_case[2][i][0],all_bad_case[2][i][1],rscd_classes[int(all_bad_case[2][i][1])]))
            cur_prop=all_bad_case[2][i][2]
            fid.write(f' the best {best_prop_num} prop is : ')
            for k in range(best_prop_num):
                fid.write(f'{cur_prop[k][0]}({rscd_classes[int(cur_prop[k][0])]}) :{cur_prop[k][1]} ')
            fid.write('\n')

    print("asphalt acc: %f, snow acc: %f"%(asphalt_acc,snow_acc))




    






