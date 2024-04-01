import os
import shutil

image_path='D:\\Work\\RSCD\\to_submit_00\\src_images'
to_path='./badcase'

if not os.path.exists(to_path):
    os.makedirs(to_path)


with open('./bad_case.txt','r') as fid:
    all_bad_case=fid.readlines()
    for bad_case in all_bad_case:
        if 'img name' not in bad_case:
            continue
        img_name=bad_case.split(' ')[2][:-1]

        ori_file=os.path.join(image_path,img_name)
        to_file=os.path.join(to_path,img_name)

        shutil.copyfile(ori_file,to_file)


