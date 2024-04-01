import torch
import RSCDModel


rscd_model=RSCDModel.RoadSurfaceModel()
rscd_model.initall(False,'/code/epoch_8.pth','/output/save_model')

# put the box to video
def DrawBoxToVideo():
    pass

def DrawBoxToImage(img,box_coordinate):
    pass

def Inference(rscd_model,img):
    pass