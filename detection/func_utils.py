import math
import os
import torch
import numpy as np
from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms_poly
import cv2
import matplotlib.pyplot as plt

from pooling import *
from select_center import Select_Center

class rthombus_target:
 def drew_rthombus(self,contour, siz=(152, 152)):
  im = np.zeros(siz, dtype="uint8")
  cv2.polylines(im, [contour.astype(np.int32)], 1, 255)
  cv2.fillPoly(im, [contour.astype(np.int32)], 255)
  # mask = im == 255
  plt.imshow(im)
  plt.show()
 def rthombus(self,box):
  # (4x2)
  c12 = (box[0, :] + box[1, :]) / 2
  c23 = (box[1, :] + box[2, :]) / 2
  c34 = (box[2, :] + box[3, :]) / 2
  c14 = (box[0, :] + box[3, :]) / 2
  contour = np.concatenate([c12, c23, c34, c14], axis=0).reshape(4, 2)
  return contour
 def restore(self,ltrb, pixel, p1):
  '''还原函数，因为坐标是h,w分布的'''
  '''输入 ltrb array(4,) or list[4] ,pixel 对应像素坐标 p1 若边框bbx(4,2) p1 (bbx[0]+bbx[1])/2'''
  top = ltrb[1]
  pixel = pixel.reshape(2, )
  pixelx1, pixely1 = pixel
  # print(pixelx1)
  p1x2, p1y2 = p1
  # print(p1x2)
  p1x2, p1y2 = p1x2, p1y2
  pixelx11, pixely11 = 0,top
  # print(p1x2, p1y2, pixelx11, pixely11)
  # print(p1x2+pixelx1,p1y2+pixely1,pixelx11+pixelx1, pixely11+pixely1)
  # b = np.array([p1x2,p1y2])
  # a = np.array([pixelx11,pixelx11])
  cosa = (pixelx11 * p1x2 + pixely11 * p1y2) / math.sqrt((pixelx11 ** 2 + pixely11 ** 2)) * math.sqrt((p1x2 ** 2 + p1y2 ** 2))
  # cosa = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
  if (cosa) > 1:
   cosa = 1.0
  elif cosa < -1:
   cosa = -1.0
  # print(cosa)


  # if (math.acos(cosa) * 180 / math.pi)>90:
  # alpha = (math.acos(cosa) * 180 / math.pi)
  sina = math.sin(math.acos(cosa))
  # print(sina,math)
  # else:
  #  alpha = 90 - (math.acos(cosa) * 180 / math.pi)
  # print(180-alpha)
  ppx = pixelx11 * cosa  +pixely11 * sina
  ppy = pixely11 * cosa -pixelx11 * sina
  # print('ppx,ppy',ppx+pixelx1,ppy+pixely1)

  # alpha = 0
  # src = np.array([[[pixelx11,pixely11]]]).astype(np.int32)
  # dst = np.array([[[p1x2,p1y2]]]).astype(np.int32)
  # A1 = cv2.getAffineTransform(src, dst)
  # print('A!',A1)
  # alpha = 140
  # alpha = 15
  # print(pixel[0],pixel[1])
  # x1, y1 = - ltrb[1], - ltrb[0]
  # # print(x1,y1)
  # x2, y2 = - ltrb[1], + ltrb[2]
  # x3, y3 = + ltrb[3], + ltrb[2]
  # x4, y4 = + ltrb[3], - ltrb[0]
  x1, y1 = -ltrb[0],ltrb[1]
  x2, y2 =ltrb[2],ltrb[1]
  x3, y3 =ltrb[2],-ltrb[3]
  x4, y4 =-ltrb[0],-ltrb[3]
  box = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(4, 2)
  # print(box)
  diu = []
  for i in range(0, 4):
   x, y = box[i]
   # print(x,y)
   # x_ = pixelx1 + x
   x_ = x * cosa - y * sina
   x_ = pixelx1 + x_
   diu.append(x_)
   # y_ = pixely1 - y
   y_ = y * cosa + x * sina
   y_ = pixely1 + y_
   diu.append(y_)
  biu = np.array(diu).reshape(4, 2)
  # print(biu)
  return biu

 def one_line(self,center, top):
  '''计算两点连成的直线坐标'''
  # print(center)
  if len(center) == 2:
      center.reshape(2,)
  cx, cy = center
  tx, ty = top
  A = ty - cy
  B = cx - tx
  C = tx * cy - cx * ty
  return [A, B, C]

 def one_vertical_line(self,para_abc, center):
  '''已知直线一般式参数以及某个点，计算这个点垂直与该直线的一般式方程参数'''
  cx, cy = center[:, 0], center[:, 1]
  A, B, C = para_abc
  A1 = B
  B1 = -A
  C1 = -(A1 * cx + B1 * cy)
  return [A1, B1, C1]

 def dis_lines(self,coord, p1, p2):
  '''已知三个点，其中coord->p1 与 coordinate->p2这两个向量的叉乘即为这三个点围城三角形面积的二倍'''
  '''该函数用于计算点到线的最短距离'''
  w = np.sqrt(np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1]))
  vec1 = p1 - coord
  vec2 = p2 - coord
  cc = np.cross(vec1, vec2)
  h = np.abs(cc) / w

  return h

 def intersection(self,para_abc1, para_abc2):
  '''求两直线的交点坐标'''
  A1, B1, C1 = para_abc1
  A2, B2, C2 = para_abc2
  y0 = (C1 * A2 - C2 * A1) / (A1 * B2 - A2 * B1)
  x0 = (C2 * B1 - C1 * B2) / (A1 * B2 - A2 * B1)
  return [x0, y0]

 def medLine(self,p1, p2, coords):
  '''求两直线的交点坐标'''
  A, B, C = self.one_line(p1, p2)
  A1, B1, C1 = self.one_vertical_line([A, B, C], coords)
  x, y = self.intersection([A, B, C], [A1, B1, C1])
  return x, y

 def parallel(self, distance, para_abc):
  A, B, C = para_abc
  # C - C1
  C1 = C - distance * math.sqrt((A ** 2 + B ** 2))
  # C2 - C
  C2 = C + distance * math.sqrt((A ** 2 + B ** 2))
  return [[A, B, C1], [A, B, C2]]
RT =rthombus_target()

class VIEW:
    #检查真实的中心点位置
    def view_center_location(self, point,results,point_list, index_list,result_list):
        if point not in point_list:
            point_list.append(point)
            index_list.append(0)
            result_list.append([results])
            # biu.append(result_list)
        else:
            index = point_list.index(point)
            index_list[index] += 1

            result_list[index].append(results)


        return point_list, index_list,result_list
    def sum_results(self,l1,l2):
        return [(l1[i]+l2[i]) for i in range(0,len(l1))]
    def append_results(self,l1,l2):
        # print(l1,l2)
        l3 = l1.append(l2)
        return l3
    def view_combine(self,im1,im2):

        combine = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)
        return combine

    def view_image(self,image):
        cv2.namedWindow('VIEW')
        cv2.imshow('VIEW',image)
        cv2.waitKey(0)

def decode_prediction(predi, dsets, args, img_id, down_ratio):

    predictions,hm = predi
    b,c,hh,hw = hm.shape
    predictions = predictions[0, :, :]
    heat1 = hm[0, :, :]
    ''''''
    heat = heat1.reshape((heat1.shape[1],heat1.shape[2],c))



    # plt.imshow(heat)
    # plt.show()
    # mask_heat = heat > 0.1
    # heat[~mask_heat] = 0
    # plt.imshow(heat)
    # plt.show()
    # for cc in range(0,heat.shape[2]):
    #     cls_heat = heat[:,:,cc]
    #     se = SegmentationMask(cls_heat)
    #     se.seg(cls_heat)

    y_max,x_max = heat1.shape[1],heat1.shape[2]

    # CV2_IM_SHOW(heat)
    # Get_Mask(heat)
    ori_image = dsets.load_image(dsets.img_ids.index(img_id))
    # imgg = ori_image.copy()
    # imgg = cv2.resize(imgg,(152,152))
    # print(dsets.img_ids.index(img_id))
    # imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    h, w, c = ori_image.shape
    # CV2_IM_SHOW(ori_image)
    he = np.uint8(255 * heat)
    he = cv2.applyColorMap(he, cv2.COLORMAP_JET)
    he = cv2.resize(he,(w,h))
    # # print(he.shape)
    #
    # super_im = cv2.addWeighted(he, 0.5, ori_image, 0.5, 0)
    # super_im = he * 0.9 + ori_image
    # CV2_IM_SHOW(ori_image)
    pts0 = {cat: [] for cat in dsets.category}
    scores0 = {cat: [] for cat in dsets.category}
    r = R()
    point_list, index_list,result_list = [], [],[]
    for pred in predictions:
        # im = np.zeros((heat.shape[0],heat.shape[1]), dtype="uint8")

        # pts0, scores0 = r.get_bbx_ltbr(pts0, scores0, pred, dsets, args, down_ratio, h, w, heat)
        pixel = np.asarray([pred[0], pred[1]], np.float32)
        pointx = np.clip(pixel[0], a_min=0, a_max=x_max-1)

        pointy = np.clip(pixel[1], a_min=0, a_max=y_max-1)
        point = np.array([pointx,pointy]).astype(np.int32).tolist()
        # print(point)
        results = [pred[2], pred[3], pred[4],pred[-1],pred[-2]]#size,theta,class,conf
        # print(results)
        # print(point,results)
        # print(pred[4])
        VIEW().view_center_location(point,results,point_list, index_list,result_list)


    im = np.zeros((heat.shape[0], heat.shape[1]))

    res = np.zeros((heat.shape[0], heat.shape[1],7))
    final_results = []
    r = R()
    f_r,pl,ind = r.result_con(result_list,point_list)

    for i in range(0,len(pl)):
        res[pl[i][1],pl[i][0]] = np.array(f_r[i])

    # print(result_list)
    SC = Select_Center(heat)
    fin_r = SC.screen_size(result_list,point_list,index_list)
    # print(fin_r)
    # p, s = SC.get_bbx(pts0, scores0, fin_r, dsets, args, down_ratio, h, w)
    for i in range(0,len(point_list)):
        im[point_list[i][1],point_list[i][0]] = index_list[i]
    #
    # Heatmap_Show().MATPLOT_SHOW(im,Multiple=False)
    # p, s = r.get_bbx(pts0, scores0, res, dsets, args, down_ratio, h, w, heat)
    p, s = get_bbx(pts0, scores0, predictions, dsets, args, down_ratio, h, w, heat)

    # return pts0, scores0 ,heat
    return p, s , heat


def non_maximum_suppression(pts, scores):
    nms_item = np.concatenate([pts[:, 0:1, 0],
                               pts[:, 0:1, 1],
                               pts[:, 1:2, 0],
                               pts[:, 1:2, 1],
                               pts[:, 2:3, 0],
                               pts[:, 2:3, 1],
                               pts[:, 3:4, 0],
                               pts[:, 3:4, 1],
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.1)
    return nms_item[keep_index]


def write_results(args,
                  model,
                  dsets,
                  down_ratio,
                  device,
                  decoder,
                  result_path,
                  print_ps=False):
    results = {cat: {img_id: [] for img_id in dsets.img_ids} for cat in dsets.category}

    for index in range(len(dsets)):
        data_dict = dsets.__getitem__(index)
        image = data_dict['image'].to(device)
        img_id = data_dict['img_id']
        image_w = data_dict['image_w']
        image_h = data_dict['image_h']
        im_size = (image_w,image_h)
        # sq = seg_question(img_size=im_size)

        with torch.no_grad():
            pr_decs = model(image)


        decoded_pts = []
        decoded_scores = []
        torch.cuda.synchronize(device)
        predictions = decoder.ctdet_decode(pr_decs)
        pts0, scores0,heat = decode_prediction(predictions, dsets, args, img_id, down_ratio)
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)
        # print(heat.shape)

        # nms
        for cat in dsets.category:
            if cat == 'background':
                continue
            pts_cat = []
            scores_cat = []
            for pts0, scores0 in zip(decoded_pts, decoded_scores):
                pts_cat.extend(pts0[cat])
                scores_cat.extend(scores0[cat])
            pts_cat = np.asarray(pts_cat, np.float32)
            scores_cat = np.asarray(scores_cat, np.float32)

            if pts_cat.shape[0]:
                # print(pts_cat)# num 4 2
                # print(scores_cat) #num 1
                nms_results = non_maximum_suppression(pts_cat, scores_cat)
                results[cat][img_id].extend(nms_results)
                cls_ind = (dsets.category.index(cat))
                cls_heat = heat[:, :, cls_ind]
                # sq.mask_seg(cls_heat,nms_results=nms_results)

        if print_ps:
            print('testing {}/{} data {}'.format(index+1, len(dsets), img_id))

    for cat in dsets.category:
        if cat == 'background':
            continue
        with open(os.path.join(result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
            for img_id in results[cat]:
                for pt in results[cat][img_id]:
                    f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))




