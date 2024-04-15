import os
import cv2
import JetourDataset

def CropToNViews(in_img_name,out_root,crop_num=4):
    if crop_num == 1:
         return

    img_name=os.path.basename(in_img_name)
    img_name=img_name.split('.')[0]
    img=cv2.imread(in_img_name)

    in_height,in_width=img.shape[0],img.shape[1]
    out_width=in_width//crop_num
    for sub_index in range(crop_num-1):
        sub_img=img[:,sub_index*out_width:(sub_index+1)*out_width]
        sub_img_name=os.path.join(out_root,f"{img_name}_{sub_index}.png")
        cv2.imwrite(sub_img_name,sub_img)
    
    sub_img=img[:,(in_width-out_width):in_width]
    sub_img_name=os.path.join(out_root,f"{img_name}_{crop_num-1}.png")

    cv2.imwrite(sub_img_name,sub_img)

if __name__ == "__main__":
    in_root_path='/data/IDTZSY/RSCD/Jetour/dataset'
    front_view='front_view_images'
    front_view_path=os.path.join(in_root_path,front_view)

    json_file='final_label.json'
    out_root='/data/IDTZSY/RSCD/Jetour/dataset/crop_images'
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    json_path=os.path.join(in_root_path,json_file)
    all_img_names=JetourDataset.GetAllImageName(json_path)

    crop_num=4
    for img_name in all_img_names:
        img_path=os.path.join(front_view_path,img_name)
        CropToNViews(img_path,out_root,crop_num)