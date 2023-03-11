import streamlit as st
import pathlib
import numpy as np
import cv2

import torch
import torchvision
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as Image

from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn.detector import detect_faces
from math import sqrt
from PIL import Image as Img

def dist(p1, p2):
    lm_x = new_lm_x
    lm_y = new_lm_y
    eye_dist = sqrt((lm_x[96]-lm_x[97])**2 +(lm_y[96]-lm_y[97])**2)
    width = sqrt((lm_x[p1]-lm_x[p2])**2 +(lm_y[p1]-lm_y[p2])**2)
    width_ratio = width/eye_dist
    width_cm = width_ratio*eye_cm
    return width_cm

def make_line(p1, p2):
    distance = dist(p1, p2)
    plt.scatter(new_lm_x[p1], new_lm_y[p1], s= 1, c = 'r')
    plt.scatter(new_lm_x[p2], new_lm_y[p2], s= 1, c = 'r')
    plt.text((new_lm_x[p1]+new_lm_x[p2])/2-150, (new_lm_y[p1]+new_lm_y[p1])/2+100, str(round(distance,2))+'cm', c = 'b')
    plt.plot([new_lm_x[p1], new_lm_x[p2]],[new_lm_y[p1], new_lm_y[p2]], c = 'b')


# streamlit image upload

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is None:
    st.write('insert your face image')

else:
    # To read file as bytes:
    #bytes_data = uploaded_file.getvalue()
    st.write(uploaded_file.name)
    st_img_path = './/streamlit_input'+'//'+uploaded_file.name
    with open(st_img_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())


        
# pytorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model_path = 'checkpoint\snapshot\checkpoint\snapshot\checkpoint_epoch_280.pth.tar'
    
    input_path = st_img_path
    output_path = 'output\\'+uploaded_file.name
    eye_cm = float(5)
    
    output_path_trigger = True
    image_show = False
    
    
    checkpoint = torch.load(model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    
    cap = cv2.VideoCapture(str(pathlib.Path(input_path)))
    
    if not (output_path_trigger is None):
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        result_path = pathlib.Path(output_path)
        result = cv2.VideoWriter(str(result_path), 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            fps, size)
    try:
        while True:
            ret, img = cap.read()
            if not ret: break
            height, width = img.shape[:2]
            bounding_boxes, landmarks = detect_faces(img)
            for box in bounding_boxes:
                x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)

                w = x2 - x1 + 1
                h = y2 - y1 + 1
                cx = x1 + w // 2
                cy = y1 + h // 2

                size = int(max([w, h]) * 1.1)
                x1 = cx - size // 2
                x2 = x1 + size
                y1 = cy - size // 2
                y2 = y1 + size

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                edx1 = max(0, -x1)
                edy1 = max(0, -y1)
                edx2 = max(0, x2 - width)
                edy2 = max(0, y2 - height)

                cropped = img[y1:y2, x1:x2]
                if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
                    cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                                 cv2.BORDER_CONSTANT, 0)

                input = cv2.resize(cropped, (112, 112))
                input = transform(input).unsqueeze(0).to(device)
                _, landmarks = pfld_backbone(input)
                pre_landmark = landmarks[0]
                pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
                    -1, 2) * [size, size] - [edx1, edy1]

                if len(list(pre_landmark)) != 98:
                    pre_lm_list = []
                    st.write('can not find face image')
                    exit()
                else:
                    pre_lm_list = pre_landmark.tolist()

                new_lm_x = []
                new_lm_y = []

                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(img, (x1 + x, y1 + y), 1, (0, 0, 255))
                    new_lm_x.append(x1+x)
                    new_lm_y.append(y1+y)

    
        cap.release()
        result.release()

        #st.write(new_lm_x)
        #st.write(new_lm_y)





        x = [i[0]*864/100 for i in pre_lm_list]
        y = [(100-i[1])*1152/100 for i in pre_lm_list]



        image = Image.imread(input_path)
        plt.imshow(image)

        #left_eye (60, 64)
        make_line(60,64)
        # right eye ()
        make_line(68,72)
        # jaw line (0, 32)
        make_line(0,32)
        # nose_line(51, 54)
        #make_line(51,54)
        # mouth line(76, 82)
        make_line(76, 82)

        plt.savefig(output_path)

        image_origin = Img.open(st_img_path)
        image_predict = Img.open(output_path)


        col1, col2 = st.columns(2)
        with col1:
            st.write('origin_image')
            st.image(image_origin, caption = 'Original image',width = 350)
        with col2:
            st.write('output_image')
            st.image(image_predict, caption = 'Result',width = 350)


        #image = Img.open('data2//'+big+'_'+small+'.jpg')
        #st.image(image, caption= big+'_'+small, width = 700)
    except:
        st.write('please use another photo, and remamber blablabla')
    