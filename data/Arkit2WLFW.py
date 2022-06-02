import os, sys
import glob
import cv2
import json
import shutil
import numpy as np
import skimage.io as ski
import skimage.transform as skt

def get_index_map(filepath):
    idxmap = dict()
    f = open(filepath, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        idxes = list(map(int, line.split('\t')))
        idxmap[idxes[1]] = idxes[0]
    return idxmap

def get_image(img_path):
    image = ski.imread(img_path)
    height, width, _ = image.shape

    screen_width, screen_height = anno['viewSize']

    ratio = screen_height/height
    resized = skt.resize(image, (screen_height, int(width*ratio)), anti_aliasing=False)

    resized_h, resized_w, _ = resized.shape
    # croped
    start_cropped = (resized_w-screen_width)//2
    cropped = resized[:, start_cropped:start_cropped+screen_width,:]
    return cropped

if __name__ == '__main__':

    img_dir = './data'
    # Grap image path and json
    json_list = glob.glob(img_dir+"/*/*.json")
    print(json_list)
    json_list = json_list[0:10]

    outDir = "./data/result"
    os.makedirs(outDir, exist_ok=True)
    
    # Get index map from arkit to wflw landmark
    idxmap = get_index_map("./data/idxmap.txt")

    for item in json_list:
        # load annotaion
        fname = item[:-5]
        imgname = fname+'.jpeg'
        with open(item) as f:
            anno = json.load(f)
        cropped = get_image(imgname)

        # filter landmark info
        x = [int(x['x']) for x in anno["ppoints"]]
        y = [int(x['y']) for x in anno["ppoints"]]

        # Get bbox size from total landmark
        bbox_x1 = min(x)
        bbox_x2 = max(x)
        bbox_y1 = min(y)
        bbox_y2 = max(y)

        # Crop face
        cropped_face = cropped[bbox_y1:bbox_y2+1, bbox_x1:bbox_x2+1]
        
        # rearrange landmarks
        x = [ a - bbox_x1 for a in x]
        y = [ a - bbox_y1 for a in y]
    
        # Add margin(10%)
        print(x[0], y[0])
        print(bbox_x1, bbox_y1) 
        
        # set attribute
        attribute = [0 for _ in range(6)] #pose, expression, illumination, make_up, occlusion, blur
        if max(anno['blendShapes'].values()) > 0.5:
            attribute[1] = 1
        print(fname)
        exit()
    
    # rotate 

    # pose expression illumination make-up occlusion blur

    # expression : load blendshape value and set 1

exit()   
# Write data
with open(os.path.join(outDir, 'test.txt'),'w') as f:
    for label in labels:
        f.writelines(label)

jaw = [221,474,475,458,459,463,469,470,460,461,462,392,921,920,922,916,1047,911,991,907,906,822,1216,887,886,1214,1043,1044,1045,1046,890,889,1032]

Lbrow = [782,608,657,767,648,768,763,764,781]
Rbrow = [199,334,209,159,349,348,329,328,335]

nose = [15,13,10,7]
ud_nose = [438,440,2,868,866]

Reye = [1101,1097,1094,1092,1089,1086,1108,1105]
Leye = [1081,1078,1076,1073,1069,1065,1062,1084]

Outer_lips = [190,100,91,21,540,549,639,572,705,29,270,279]
Inner_lips = [404,255,24,690,834,708,26,273]

part_lst = [jaw, Lbrow, Rbrow, nose, ud_nose, Reye, Leye, Outer_lips, Inner_lips]
part_color = ['blue', 'darkgreen', 'green', 'black', 'gold', 'deeppink', 'hotpink', 'cyan', 'red']

# Open list file
# Write info
