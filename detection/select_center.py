import os
import math
import cv2
import numpy as np
from tools import Heatmap_Show,Decoder_REG_Targets


'''这个类用于筛选出预测的中心点'''

'''数值下降的方式'''
class Select_Center:
    def __init__(self,heatmap):
        self.heatmap = heatmap
        self.im = np.zeros((heatmap.shape[0], heatmap.shape[1]))
        self.Decoder_reg = Decoder_REG_Targets()


    def screen_size(self,result_list,point_list,index_list):
        '''去掉一个最大值，去掉一个最小值，或者之间取平均'''
        re_l = []
        for re_ind in range(0,len(result_list)):
            re_np = np.array(result_list[re_ind])
            # print(re_np.shape)
            if re_np.shape[0] > 2:
                w = (np.sum(re_np[:,0])-np.max(re_np[:,0])-np.min(re_np[:,0]))/(re_np.shape[0]-2)
                h = (np.sum(re_np[:,1])-np.max(re_np[:,1])-np.min(re_np[:,1]))/(re_np.shape[0]-2)
                theta = (np.sum(re_np[:,2])-np.max(re_np[:,2])-np.min(re_np[:,2]))/(re_np.shape[0]-2)
                cls = (np.sum(re_np[:, 3])) / re_np.shape[0]
                conf = (np.sum(re_np[:, 4])) / re_np.shape[0]
            else:
                w = (np.sum(re_np[:, 0])) / re_np.shape[0]
                h = (np.sum(re_np[:, 1])) / re_np.shape[0]
                theta = (np.sum(re_np[:, 2])) / re_np.shape[0]
                cls = (np.sum(re_np[:, 3])) / re_np.shape[0]
                conf = (np.sum(re_np[:, 4])) / re_np.shape[0]
            point_x,point_y = point_list[re_ind]
            num_ind = index_list[re_ind]
            # if num_ind > 0:
            re_l.append([point_x,point_y,w,h,theta,cls,conf,num_ind])
        for i in re_l:
            x,y = int(i[1]),int(i[0])
            self.im[x,y] = i[-1]

            #

        re_l_np = np.array(re_l)
        final_results = []
        # re_l_np_sort = (re_l_np[np.argsort(re_l_np[:, -1])][::-1])
        # c = []
        # for a in (re_l_np_sort.tolist()[1:]):
        #     center = re_l_np_sort.tolist()[0]
        #     # a = bb[0]
        #     b = center
        #     v = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        #     if v>(center[3]/2)**2:
        #         c.append(a)
        # for i in c:
        #     print(c)
        # Heatmap_Show().MATPLOT_SHOW(self.im, Multiple=False)
        self.screen_points(final_results,re_l_np)
        # print(final_results)
        # for i in range(0,len(final_results)):
        #     hm_x = final_results[i][1]
        #     hm_y = final_results[i][0]
        #     self.heatmap[int(hm_x),int(hm_y)] = 2
        #
        # Heatmap_Show().MATPLOT_SHOW(self.heatmap,Multiple=False)
        return final_results


    def screen_points(self,final_results,re_l_np):
        # re_l_np = np.array(re_l)
        # print(re_l_np[0].tolist())
        if (re_l_np.shape[0]==0):
            return
        else:
            # print(re_l_np)
            re_l_np_sort = (re_l_np[np.argsort(re_l_np[:, -1])][::-1])
            seeds = re_l_np_sort[0, :]
            mask = np.array(
                [self.in_roundness(re_l_np_sort[i, 0:2], seeds[0:2], seeds[3]) for i in
                 range(0, re_l_np_sort.shape[0])])
            final_results.append(seeds.tolist())
            return self.screen_points(final_results, re_l_np_sort[~mask])

    def in_roundness(self,point,center,radius):
        '''input: point坐标点 center 圆心，指代图中index最大的坐标点的位置，radius代表园的半径，使用最短边'''
        '''output: bool'''
        point_x,point_y = point
        center_x,center_y = center
        P_R = (point_x-center_x)**2 + (point_y-center_y)**2
        R_R = (radius)**2
        # print(point,center)
        # print(P_R,R_R)
        if P_R < R_R or P_R == R_R:
            return True
        else:
            return False

    def get_bbx(self, pts0, scores0, result, dsets, args, down_ratio, h, w):
        # pts0 = {cat: [] for cat in dsets.category}
        # scores0 = {cat: [] for cat in dsets.category}

        # Seg_Results().segment(self.heatmap)
        for i in range(0, len(result)):
            center_x, center_y, all_w, all_h, all_theta, clse, score ,_= result[i]

            cbbx= self.Decoder_reg.decoder_reg((center_x, center_y), (all_w, all_h), all_theta)


            cbbx[:, 0] = cbbx[:, 0] * down_ratio / args.input_w * w
            cbbx[:, 1] = cbbx[:, 1] * down_ratio / args.input_h * h
            # print(score)
            if score > 0.13:
                pts0[dsets.category[int(clse)]].append(cbbx)
                scores0[dsets.category[int(clse)]].append(score)
            else:
                continue


        return pts0, scores0

