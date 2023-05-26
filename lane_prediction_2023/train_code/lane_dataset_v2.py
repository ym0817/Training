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
import math
import pandas as pd
import csv
import random
import glob



class Dataset(data.Dataset):
    def __init__(self, data_root):

        self.data_root = data_root
        self.frame_step = 5       #   每隔5帧预测一帧
        self.lane_cnt = 2         #   每八个目标构建一帧
        self.points_num = 16      #   采样点数

        self.train_data_list = []
        self.train_label_list = []

        file_paths = glob.glob(self.data_root + '/*csv')
        data_size_list = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            labels = list(df.columns.values)
            print(labels)
            lines = df.values.tolist()

            single_temp = self.get_traintype(lines)
            for single in single_temp:     # single为连续 self.frame_step * self.lane_cnt 帧
                # self.all_temp_frames.append(single)
                # single_num = len(single) //  self.lane_cnt
                if len(single) != self.frame_step *  self.lane_cnt :
                    continue
                else:
                    im, label = [], []
                    left_sidelist, right_sidelist = [], []
                    left_lines, right_lines = [], []
                    for i in range(self.frame_step):
                        left_line = single[2 * i]
                        right_line = single[2 * i + 1]
                        left_sidelist.append(left_line[1])
                        right_sidelist.append(right_line[1])
                        left_lines.append(left_line)
                        right_lines.append(right_line)

                    if left_sidelist.count(1.0)  ==  self.frame_step or right_sidelist.count(2.0) == self.frame_step :
                        for ii in range(self.frame_step):
                            l_side = left_lines[ii][1]
                            r_side = right_lines[ii][1]
                            if l_side == 0.0 :
                                lxy_list = [ 0.0 for ln in range(self.points_num * 2)]
                            else:
                                lxy_list = self.ana_line(left_lines[ii], self.points_num)

                            if r_side == 0.0 :
                                rxy_list = [0.0 for rn in range(self.points_num * 2)]
                            else:
                                rxy_list = self.ana_line(right_lines[ii], self.points_num)

                            if ii != self.frame_step - 1:
                                im.append(lxy_list)
                                im.append(rxy_list)
                            else:
                                label.append(lxy_list)
                                label.append(rxy_list)
                    else:
                        continue

                    np_label = []
                    for la in label:
                        for i in la:
                            np_label.append(i)

                    np_im = [np.array(i) for i in im]
                    np_im_st = np.stack(np_im, 0)
                    im_ex = np.expand_dims(np_im_st, 0)

                    label_ex = np.array(np_label)
                    self.train_data_list.append(im_ex)
                    self.train_label_list.append(label_ex)

            data_size_list.append(len(self.train_data_list))
            if len(data_size_list) == 1:
                data_size = len(self.train_data_list)
                print("first data size ", file_path, data_size_list[0])
            else:
            #     data_size = len(self.train_data_list)-data_size_list[-1]
                print("other data size ", file_path, data_size_list[-1]-data_size_list[-2])
        print('total data size', len(self.train_data_list))





    def get_traintype(self, single_lines):
        frames_num = len(single_lines) // self.lane_cnt
        all_frames = [single_lines[i * self.lane_cnt : (i + 1) * self.lane_cnt] for i in range(frames_num)]

        single_all_temp_frames = []
        for j in range(0, len(all_frames)-self.frame_step):
            temp_frame_step = all_frames[ j*self.frame_step : (j+1)*self.frame_step]
            temp = [ aa for tt in temp_frame_step for aa in tt ]
            single_all_temp_frames.append(temp)
        return single_all_temp_frames



    # def ana_line(self, line, k_step):
    #
    #     start = line[2]
    #     end = line[3]
    #
    #     c0 = line[4]  # (-10.0, 10.0)
    #     c1 = line[5]  # (-0.357, 0.357)
    #     c2 = line[6]  # (-0.032,0.032)
    #     c3 = line[7]  # (-0.000122, 0.000122)
    #
    #     en_st = end - start
    #     x_list, z_list, headimg = [], [], []
    #     xy = []
    #
    #     for i in np.arange(0, en_st, en_st / k_step):
    #         temp_z = start + i
    #         x = c3 * math.pow(temp_z, 3) + c2 * math.pow(temp_z, 2) + c1 * temp_z + c0
    #         x_list.append(x)
    #         z_list.append(temp_z)
    #
    #         headimg_angle = 3 * c3 * math.pow(temp_z, 2) + 2 * c2 * temp_z + c1
    #         headimg.append(headimg_angle)
    #     xy = x_list + z_list
    #
    #     return xy, headimg


    def ana_line(self, line, k_step):

        start = line[2]
        end = line[3]

        c0 = line[4]  # (-10.0, 10.0)
        c1 = line[5]  # (-0.357, 0.357)
        c2 = line[6]  # (-0.032,0.032)
        c3 = line[7]  # (-0.000122, 0.000122)

        z_list = np.linspace(start, end, k_step).tolist()
        x_list = [c3 * math.pow(temp_z, 3) + c2 * math.pow(temp_z, 2) + c1 * temp_z + c0 for temp_z in z_list]
        xy = x_list + z_list

        return xy




    def __getitem__(self, index):
        dataim = self.train_data_list[index]
        target = self.train_label_list[index]
        # print ('aaaa', len(dataim))
        #
        tensor_im = torch.from_numpy(dataim).float()    #.to(torch.float32)
        tensor_target = torch.from_numpy(target).float()     #.to(torch.float32)


        return tensor_im, tensor_target

    def __len__(self):
        return len(self.train_data_list)










if __name__ == '__main__':
    train_data = Dataset("lane_csvfiles")

    trainloader = data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    for i, (data, label) in enumerate(trainloader):
        # print (label)
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.5, 0.5, 0.5])
        # img += np.array([0.5, 0.5, 0.5])
        img *= 255
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        cv2.imshow('img', img)
        if cv2.waitKey(200):
            continue
    
