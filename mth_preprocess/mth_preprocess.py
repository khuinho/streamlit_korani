# From data Imagefolder process PFLD to prepare WLFW type dataset
import os, sys
sys.path.append('.')
sys.path.append('..')
import argparse
import pathlib
import glob
import cv2
import json
import numpy as np
import skimage.io as ski
import skimage.transform as skt
import matplotlib.pyplot as plt


import pandas as pd
import imageio
import shutil

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import matplotlib.image as img
from tqdm import tqdm
from PIL import Image as image

from pfld.utils import calculate_pitch_yaw_roll
from dataset.datasets import WLFWDatasets
from models.pfld import PFLDInference

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rootdir = '.'
# save_path = './/data//data//'
# mth_idxmap_path = 'data//mth_idxmap.txt'
# print('mth preprocessing:\n')
# mth_preprocess(rootdir, save_path, mth_idxmap_path)

def mth_preprocess(rootdir, save_path, mth_idxmap_path):
    num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
           66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]
    name = ['FACIAL_R_Sideburn3', 'FACIAL_R_Sideburn6', 'FACIAL_R_JawRecess', 'none0', 'none1', 'FACIAL_R_JawBulge', 'none2', 'FACIAL_R_12IPV_Jawline5',
            'none3', 'none4', 'FACIAL_R_12IPV_Jawline4', 'none5', 'none6', 'FACIAL_R_12IPV_ChinS4', 'none7', 'none8', 'FACIAL_C_12IPV_Chin3', 'none9', 'none10',
            'FACIAL_L_12IPV_ChinS4', 'none11', 'none12', 'FACIAL_L_12IPV_Jawline4', 'none13', 'none14', 'FACIAL_L_12IPV_Jawline5', 'none15', 'FACIAL_L_JawBulge',
            'none16', 'none17', 'FACIAL_L_JawRecess', 'FACIAL_L_Sideburn6', 'FACIAL_L_Sideburn3', 'FACIAL_R_12IPV_ForeheadOut31', 'FACIAL_R_12IPV_ForeheadOut28',
            'FACIAL_R_12IPV_ForeheadMid21', 'FACIAL_R_12IPV_ForeheadMid17', 'FACIAL_R_12IPV_ForeheadIn7', 'FACIAL_R_12IPV_ForeheadIn13', 'FACIAL_R_12IPV_ForeheadMid18',
            'FACIAL_R_12IPV_ForeheadMid22', 'FACIAL_R_12IPV_ForeheadOut29', 'FACIAL_L_12IPV_ForeheadIn7', 'FACIAL_L_12IPV_ForeheadMid17', 'FACIAL_L_12IPV_ForeheadMid21',
            'FACIAL_L_12IPV_ForeheadOut28', 'FACIAL_L_12IPV_ForeheadOut31', 'FACIAL_L_12IPV_ForeheadOut29', 'FACIAL_L_12IPV_ForeheadMid22', 'FACIAL_L_12IPV_ForeheadMid18',
            'FACIAL_L_12IPV_ForeheadIn13', 'FACIAL_C_12IPV_NoseBridge2', 'FACIAL_C_12IPV_NoseUpper2', 'FACIAL_C_12IPV_NoseTip1', 'FACIAL_C_NoseTip', 'FACIAL_R_12IPV_Nostril11',
            'FACIAL_R_12IPV_Nostril12', 'FACIAL_C_12IPV_NoseL1', 'FACIAL_L_12IPV_Nostril12', 'FACIAL_L_12IPV_Nostril11', 'FACIAL_R_EyeCornerOuter', 'FACIAL_R_EyelashesUpperA3',
            'FACIAL_R_EyelashesUpperA2', 'FACIAL_R_EyelashesUpperA1', 'FACIAL_R_EyeCornerInner', 'FACIAL_R_EyelidLowerA1', 'FACIAL_R_EyelidLowerA2', 'FACIAL_R_EyelidLowerA3',
            'FACIAL_L_EyeCornerInner', 'FACIAL_L_EyelashesUpperA1', 'FACIAL_L_EyelashesUpperA2', 'FACIAL_L_EyelashesUpperA3', 'FACIAL_L_EyeCornerOuter', 'FACIAL_L_EyelidLowerA3',
            'FACIAL_L_EyelidLowerA2', 'FACIAL_L_EyelidLowerA1', 'FACIAL_R_12IPV_LipCorner2', 'FACIAL_R_12IPV_LipUpper17', 'FACIAL_R_LipUpper', 'FACIAL_C_12IPV_LipUpperSkin2',
            'FACIAL_L_LipUpper', 'FACIAL_L_12IPV_LipUpper17', 'FACIAL_L_12IPV_LipCorner2', 'FACIAL_L_12IPV_LipLower17', 'FACIAL_L_12IPV_LipLower9', 'FACIAL_C_12IPV_LipLowerSkin1',
            'FACIAL_R_12IPV_LipLower9', 'FACIAL_R_12IPV_LipLower17', 'FACIAL_R_12IPV_LipCorner1', 'FACIAL_R_12IPV_LipUpper14', 'FACIAL_C_LipUpper', 'FACIAL_L_12IPV_LipUpper14', 
            'FACIAL_L_12IPV_LipCorner1', 'FACIAL_L_12IPV_LipLower14', 'FACIAL_C_LipLower', 'FACIAL_R_12IPV_LipLower14', 'FACIAL_R_Pupil', 'FACIAL_L_Pupil']
    num_name = pd.DataFrame({'num': tuple(num), 'name': tuple(name)})


    for path in tqdm (glob.glob(f'{rootdir}/*/*/*/*')):
        if str(path)[-3:] == 'png':
        
            img_name = path.split('\\')[4]
            img = image.open(path)
            img_resize = img.resize((112,112))
            img_jpg = img_resize.convert('RGB')
            img_jpg.save('{}//{}jpeg'.format(save_path, img_name[:-3]))
            #img_resize.save('{}//{}png'.format(save_path, img_name[:-3]))
            #print('{}//{}jpeg'.format(save_path, img_name[:-3]))
    
        elif str(path)[-4:] == 'json': 
            shutil.copy(path, save_path)


    values = {}
    json_to_dict = {}

    cnt = 0
    file_names_list = os.listdir(save_path)
    file_name_list = []
    for file_names in file_names_list:
        file_name_list.append(file_names.split('.')[0])


    f = open(mth_idxmap_path, 'w')
    print('mth_idxmap.txt making')
 
    for file_name in tqdm(file_name_list):
        #img_path = os.path.join(save_path, file_name+'.jpeg')

        #img_x , img_y = list(imageio.imread(img_path).shape)[0], list(imageio.imread(img_path).shape)[1]    
        data_json_path = os.path.join(save_path, file_name+'.json')

        with open(data_json_path, 'r') as f2:
            data_json = json.load(f2)

        for landmark in data_json['BonePosition']:
            json_to_dict[landmark['BoneName']] = landmark['ScreenPosition']

        for name in list(num_name['name']):
            if name[0] == 'n':
                values[name] = 'none'
                cnt +=1
            else:
                values[name] = json_to_dict[name]

        stack = 0

        for idx, key_and_value in enumerate(list(values.items())):
            key, value = key_and_value
            if key[0] == 'n':
                stack +=1

            else:
                if stack == 1:
                    values[list(values.keys())[idx-1]] = [(values[list(values.keys())[idx-2]][0]+values[list(values.keys())[idx]][0])/2, 
                                                    (values[list(values.keys())[idx-2]][1]+values[list(values.keys())[idx]][1])/2]

                elif stack == 2:
                    values[list(values.keys())[idx-2]] = [values[list(values.keys())[idx-3]][0]+(values[list(values.keys())[idx]][0]- values[list(values.keys())[idx-3]][0])/3, 
                                                          values[list(values.keys())[idx-3]][1]+(values[list(values.keys())[idx]][1]- values[list(values.keys())[idx-3]][1])/3]

                    values[list(values.keys())[idx-1]] = [values[list(values.keys())[idx-3]][0]+(values[list(values.keys())[idx]][0]- values[list(values.keys())[idx-3]][0])*2/3, 
                                                          values[list(values.keys())[idx-3]][1]+(values[list(values.keys())[idx]][1]- values[list(values.keys())[idx-3]][1])*2/3]
                stack = 0

    
        f.write('{}'.format(save_path))
        f.write(file_name)
        f.write('.png ')

        for name in values.keys():
            f.write(str(values[name][0]/512))
            f.write(' ')
            f.write(str(values[name][1]/512))
            f.write(' ')
        attribute = '0 0 0 0 0 0 '
        for blendshapetype in data_json["ARKit"]:
            if float(blendshapetype["Value"]) >= 0.5:
                attribute = '0 1 0 0 0 0 '
        f.write(attribute)
        #print(attribute)
        f.write(str(data_json["HeadRotation"][0]))
        f.write(' ')
        f.write(str(data_json["HeadRotation"][1]))
        f.write(' ')
        f.write(str(data_json["HeadRotation"][2]))
        f.write('\n')

    f.close()
    print('mth_idxmap.txt done!')

def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

def draw_example(img, lmks, img_size, imgname):
    # Draw image
    fig = plt.figure()
    imgplot = plt.imshow(img[:,:,::-1])

    # draw landmark
    plt.scatter(lmks[:,0]*img_size, lmks[:,1]*img_size, c='blue', s=1)

    # save image
    plt.savefig(imgname+".png")
    plt.close()

class ImageData():
    def __init__(self, img, landmark, box, flag, network=None, transform=None, image_size=112):
        self.image_size = image_size
        #0-195: landmark 坐标点  196-199: bbox 坐标点;
        #200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        #206: 图片名称
        
        self.landmark = landmark
        self.box = box
        # flag = list(map(int, line[200:206]))
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        self.raw_img = img # json data path
        self.img = None
        self.network = network
        self.transform = transform

        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def set_lmks(self, landmark):
        self.landmark = landmark
    
    def set_bbox(self, box):
        self.box = box

    def set_flag(self, flag):
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]

    def get_img(self):
        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        img = self.raw_img
        boxsize = int(np.max(wh)*1.4)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark+0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        
        # self.raw_img = imgT
        return imgT

    def load_data(self, is_train, repeat, img_name, mirror=None, draw=False):
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        xy = np.min(self.landmark, axis=0).astype(np.float) 
        zz = np.max(self.landmark, axis=0).astype(np.float)
        wh = zz - xy + 1
        #wh = np.array([int(wh[0]), int(wh[1])]) 
        
        center = (xy + wh/2).astype(np.int32)
        img = self.raw_img
        boxsize = int(np.max(wh)*1.2)

        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)



        imgT = img[y1:y2, x1:x2]
        
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark+0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        # Network input
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        
        
        # transform to tensor
        t_resized = self.transform(imgT)
        t_resized = torch.unsqueeze(t_resized, 0)
        t_resized = t_resized.to(device)
        
        # get lmks
        _, pfld_lmk = self.network(t_resized)
        pfld_lmk = pfld_lmk.detach().cpu().numpy()
        pfld_lmk = pfld_lmk.reshape(-1, 2)  # landmark
        
        if draw:
            draw_example(imgT, pfld_lmk, self.image_size, "imgT.png")

        # save cropped image
        self.landmarks.append(pfld_lmk)

        # resize output lmk to pixel level fit input image
        # add input image starting point to output lmk
        landmark = pfld_lmk * boxsize + xy
        
        # Update landmark
        self.landmark = landmark
        
        if draw:
            draw_example(img, landmark, 1.0, "img")

        # raw input image landmark to 0~1 network input image level
        landmark = pfld_lmk # (self.landmark - xy)/boxsize
        
        if (landmark < 0).all() or (landmark > 1).all():
            print(img_name, "is Clipped")

            print("original:", str(landmark) + str([dx, dy]))
            np.clip(landmark, 0, 1, out=landmark)
            print("clipped:", str(landmark) + str([dx, dy]))

        # assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        # assert (landmark <= 1).all(), str(landmark) + str([dx, dy])

        self.imgs.append(imgT)
        # self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx,cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1]*1.1), int(img.shape[0]*1.1)))
                
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:,0] = 1 - landmark[:,0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)

                if draw:
                    draw_example(imgT, landmark, self.image_size, "img_"+str(len(self.imgs)))

                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = os.path.join(path, prefix+'_'+str(i)+'.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str,lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str, attributes_str, euler_angles_str)

            labels.append(label)
        return labels

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

    ##screen_width, screen_height = anno['viewSize']
    screen_width, screen_height = [112,112]

    ratio = screen_height/height #0.6
    resized = skt.resize(image, (screen_height, int(width*ratio)), anti_aliasing=False)

    resized_h, resized_w, _ = resized.shape
    # croped
    start_cropped = (resized_w-screen_width)//2
    cropped = (resized[:, start_cropped:start_cropped+screen_width,:] * 255).astype(np.uint8)

    return cropped[:,:,::-1]

def parse_args():
    parser = argparse.ArgumentParser(description='Help for data preprocessing arguments')
    parser.add_argument('--model_path',
                        default="./checkpoint/checkpoint/checkpoint_pfld.pth.tar",
                        type=str)
    parser.add_argument('--imgs_path',
                        default="D:\pfld_data\pfld_data\data\data",
                        type=str)
    parser.add_argument('--output',
                        default="D:\pfld_data\pfld_data\data//result",
                        type=str)
    parser.add_argument('--train', default=True, action='store_true')
                        
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    ##rootdir = '.'
    ##save_path = './/data//data//'
    mth_idxmap_path = 'D://pfld_data//pfld_data//data//mth_idxmap.txt'
    print('mth preprocessing:\n')
    #mth_preprocess(rootdir, save_path, mth_idxmap_path)
    
    # Set data folder
    img_dir = pathlib.Path(args.imgs_path)
    # videodir = pathlib.Path('*')
    filepath = pathlib.Path('*.json')
    # input_dirs = img_dir/videodir/filepath
    input_dirs = img_dir/filepath
    # Grap image path and json
    json_list = glob.glob(str(input_dirs))
    # json_list = json_list[0:10]
    outDir = pathlib.Path(args.output)
    
    os.makedirs(outDir, exist_ok=True)
    save_img = os.path.join(outDir, 'imgs')
    if not os.path.exists(save_img):
        os.mkdir(save_img)
    # Get index map from arkit to wflw landmark
    idxmap_path = str(pathlib.Path("data/mth_idxmap.txt"))
    ##idxmap = get_index_map(idxmap_path)

    labels = []
    # is_train = True
    Mirror_file = None

    # Load pfld models
    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    
    f = open(mth_idxmap_path, 'r')
    idxmatch = {}
    for line in f:
        line_list = line.split(' ')
        idxmatch[line_list[0].split('/')[-1][:-5]] = {'x': line_list[1:197][::2],'y':  line_list[1:197][1::2], 'att': line_list[197:203]} 
    f.close()


    for i, item in enumerate(json_list):
        # load annotaion
        fname = item[:-5]
        img_name = fname+'.jpeg'
        with open(item) as f:
            anno = json.load(f)
        resized = get_image(img_name) # Fix viewSize image(range of arkit)
        
        # filter landmark info
        ##x = [x['x'] for x in anno["ppoints"]]
        ##y = [x['y'] for x in anno["ppoints"]]
        fname2 = fname.split('\\')[-1]

        x =[float(j) for j in list(idxmatch[fname2]['x'])]
        y = [float(k) for k in list(idxmatch[fname2]['y'])]
        lmks = np.array([x,y], dtype=np.float32).T

        # make wflw landmark from arkit data
        #TODO update size 96 -> 98 for iris value
        wlfw_lmks = np.zeros((98,2), np.float32)
        ##for k,v in idxmap.items():
        ##    wlfw_lmks[v] = lmks[k]
    
        for idx, sx in enumerate(x):
            wlfw_lmks[idx] = np.array((float(sx)*112,float(y[idx])*112))

        # Get bbox size from total landmark
        bbox_x1 = int(min(x)*112)
        bbox_x2 = int(max(x)*112)
        bbox_y1 = int(min(y)*112)
        bbox_y2 = int(max(y)*112)
        
        bbox = np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2], dtype=np.int32)

        # set attribute - pose expression illumination make-up occlusion blur
        attribute = [int(l) for l in idxmatch[fname2]['att']]
        # expression : load blendshape value and set 1
        # Set expression
        #TODO Reset expression from json file <- Update key
        ##if 'face_expression' in anno.keys(): #max(anno['blendShapes'].values()) > 0.5:
        ##    if anno['face_expression'] <= 1:                
        ##        attribute[1] = anno['face_expression']
                
        # item: image file path
        # 96*2
        Img = ImageData(resized, wlfw_lmks, bbox, attribute, network=pfld_backbone, transform=transform)
        #Img.load_data(args.train, 10, img_name, Mirror_file, draw=False)
        Img.load_data(True, 10, img_name, Mirror_file, draw=False)
        _, filename = os.path.split(img_name)
        filename, _ = os.path.splitext(filename)
        label_txt = Img.save_data(save_img, str(i)+'_' + filename)
        labels.append(label_txt)
        if ((i + 1) % 1) == 0:
            print('file: {}/{}'.format(i+1, len(json_list)))

    # Write data
    with open(os.path.join(outDir, 'list.txt'),'w') as f:
        for label in labels:
            f.writelines(label)