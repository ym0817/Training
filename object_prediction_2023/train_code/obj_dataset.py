import os
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import torchvision
import cv2
import sys
import random
from PIL import Image
import glob
import pandas as pd
import csv
import random



class Dataset(data.Dataset):
    def __init__(self, data_root):

        self.data_root = data_root
        self.frame_step = 5       #   每隔5帧预测一帧
        self.obj_cnt = 12         #   每12个目标构建一帧

        self.train_data_list = []
        self.train_label_list = []
        data_size_list = []
        file_paths = glob.glob(self.data_root + '/*csv')
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            labels = list(df.columns.values)
            # print(labels)
            lines = df.values.tolist()

            single_temp = self.get_traintype(lines)

            for single in single_temp:  # single为连续 self.frame_step * self.lane_cnt 帧
                if len(single) != self.frame_step * self.obj_cnt:
                    continue
                else:
                    obj_list = [per_line[1] for per_line in single]
                    obj_set = list(set(obj_list))

                    for obj_id in obj_set:
                        state_list = []
                        for per_line in single:
                            if per_line[1] == obj_id and obj_id !=  0.0:
                                x = per_line[3]
                                y = per_line[4]
                                vx = per_line[5]
                                vy = per_line[6]
                                ay = per_line[7]
                                v = per_line[8]
                                thta = per_line[9]
                                thta_rate = per_line[10]

                                state = [x, y, vx, vy, ay, v, thta, thta_rate]
                                state_list.append(state)

                        if len(state_list) != self.frame_step:
                            continue
                        else:
                            np_im = [np.array(i) for i in state_list[:-1]]

                            im = np.stack(np_im, 0)
                            im_ex = np.expand_dims(im,0)

                            label = np.array(state_list[-1])   #/ 1000

                            self.train_data_list.append(im_ex)
                            self.train_label_list.append(label)

            data_size_list.append(len(self.train_data_list))
            if len(data_size_list) == 1:
                data_size = len(self.train_data_list)
                print("first data size ", file_path, data_size_list[0])
            else:
                #     data_size = len(self.train_data_list)-data_size_list[-1]
                print("other data size ", file_path, data_size_list[-1] - data_size_list[-2])
        print('total data size', len(self.train_data_list))



    def get_traintype(self, single_lines):
        frames_num = len(single_lines) // self.obj_cnt
        all_frames = [single_lines[i * self.obj_cnt: (i + 1) * self.obj_cnt] for i in range(frames_num)]

        single_all_temp_frames = []
        for j in range(0, len(all_frames) - self.frame_step):
            temp_frame_step = all_frames[j * self.frame_step: (j + 1) * self.frame_step]
            temp = [aa for tt in temp_frame_step for aa in tt]
            single_all_temp_frames.append(temp)
        return single_all_temp_frames




    def __getitem__(self, index):
        dataim = self.train_data_list[index]
        target = self.train_label_list[index]
        # print ('aaaa', len(dataim))

        tensor_im = torch.from_numpy(dataim).float()    #.to(torch.float32)

        tensor_target = torch.from_numpy(target) .float()    #.to(torch.float32


    
        return tensor_im, tensor_target

    def __len__(self):
        return len(self.train_data_list)




if __name__ == '__main__':
    train_data = Dataset("/media/ym/DATA1/Arhud_Prediction/DataProcess/object_csvfiles")

    trainloader = data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        img *= np.array([0.5, 0.5, 0.5])
        img += np.array([0.5, 0.5, 0.5])
        img *= 255
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        cv2.imshow('img', img)
        if cv2.waitKey(200):
            continue
    
