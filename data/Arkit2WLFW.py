import os, sys
import glob
import cv2
import json
import shutil
import numpy as np
import skimage.io as ski
import skimage.transform as skt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from pfld.utils import calculate_pitch_yaw_roll

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

class ImageDate():
    def __init__(self, img, landmark, box, flag, image_size=112):
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

        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def load_data(self, is_train, repeat, mirror=None):
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

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
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy)/boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
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
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (96, 2)
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

    screen_width, screen_height = anno['viewSize']

    ratio = screen_height/height
    resized = skt.resize(image, (screen_height, int(width*ratio)), anti_aliasing=False)

    resized_h, resized_w, _ = resized.shape
    # croped
    start_cropped = (resized_w-screen_width)//2
    cropped = (resized[:, start_cropped:start_cropped+screen_width,:] * 255).astype(np.uint8)

    return cropped[:,:,::-1]

if __name__ == '__main__':

    img_dir = './data'
    # Grap image path and json
    json_list = glob.glob(img_dir+"/*/*.json")
    json_list = json_list[0:10]

    outDir = "./data/result"
    os.makedirs(outDir, exist_ok=True)
    save_img = os.path.join(outDir, 'imgs')
    if not os.path.exists(save_img):
        os.mkdir(save_img)
    # Get index map from arkit to wflw landmark
    idxmap = get_index_map("./data/idxmap.txt")
    labels = []
    is_train = True
    Mirror_file = None

    for i, item in enumerate(json_list):
        # load annotaion
        fname = item[:-5]
        img_name = fname+'.jpeg'
        with open(item) as f:
            anno = json.load(f)
        resized = get_image(img_name) # Fix viewSize image(range of arkit)
        a = cv2.imread(img_name)
        print(type(resized), resized.shape)
        print(type(a), a.shape)
        # filter landmark info
        x = [x['x'] for x in anno["ppoints"]]
        y = [x['y'] for x in anno["ppoints"]]
        lmks = np.array([x,y], dtype=np.float32).T

        # make wflw landmark from arkit data
        #TODO update size 96 -> 98 for iris value
        wlfw_lmks = np.zeros((96,2), np.float32)
        for k,v in idxmap.items():
            wlfw_lmks[v] = lmks[k]
            
        #TODO Get iris        

        # Get bbox size from total landmark
        bbox_x1 = int(min(x))
        bbox_x2 = int(max(x))
        bbox_y1 = int(min(y))
        bbox_y2 = int(max(y))
        
        bbox = np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2], dtype=np.int32)

        # set attribute - pose expression illumination make-up occlusion blur
        attribute = [0 for _ in range(6)]
        # expression : load blendshape value and set 1
        if max(anno['blendShapes'].values()) > 0.5:
            attribute[1] = 1
        
        # item: image file path
        Img = ImageDate(resized, wlfw_lmks, bbox, attribute)
        Img.load_data(is_train, 10, Mirror_file)
        _, filename = os.path.split(img_name)
        filename, _ = os.path.splitext(filename)
        label_txt = Img.save_data(save_img, str(i)+'_' + filename)
        labels.append(label_txt)
        if ((i + 1) % 1) == 0:
            print('file: {}/{}'.format(i+1, len(json_list)))

    # Write data
    with open(os.path.join(outDir, 'test.txt'),'w') as f:
        for label in labels:
            f.writelines(label)