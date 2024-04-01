import cv2
import os
import numpy as np

# all in order W x H
img_size=[1920,1080]
top_start_height=121

left_part_start_width=0
left_part_end_width=596

right_part_start_width=600
right_part_end_width=1920

right_road_region_tl=[1121,510]
right_road_region_tr=[1428,510]
right_road_region_bl=[745,850]
right_road_region_br=[1818,850]

left_road_region_tl=[213,top_start_height]
left_road_region_tr=[382,top_start_height]
left_road_region_bl=[213,381]
left_road_region_br=[382,381]


def GetRightFrontRoad(img):
    right_part=img[top_start_height:,right_part_start_width:right_part_end_width]

    front_view=img[]
    
    road_img_width=right_road_region_br[0]-right_road_region_bl[0]+1
    road_img_height=right_road_region_bl[1]-right_road_region_tl[1]+1

    slop_l=(right_road_region_bl[0]-right_road_region_tl[0])
    

if __name__ == "__main__":
    pass