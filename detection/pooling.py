import numpy as np
import cv2
from tools import Decoder_REG_Targets, HEAT_SIZE,Encoder_LTBR
import matplotlib.pyplot as plt
from tools import Heatmap_Show
import math


# 周围填充
def zero_padding(in_array, padding_size=1):
    rows, cols = in_array.shape
    padding_array = np.zeros([rows + 2 * padding_size, cols + 2 * padding_size])
    padding_array[padding_size:rows + padding_size, padding_size:cols + padding_size] = in_array
    return padding_array


# 末尾填充，看情况

def end_zero_padding(in_array, padding_size=1):
    rows, cols = in_array.shape
    padding_array = np.zeros([rows + padding_size, cols + padding_size])
    padding_array[0:rows, 0:cols] = in_array
    return padding_array


# 计算中位数
# 需要将每个点的数据写成list

class R:
    def __init__(self):
        self.Decoder_reg = Decoder_REG_Targets()
        self.D_ltrb = Encoder_LTBR()


    def calcMedian(self, data):
        # print(data)
        if len(data) % 2 == 0:
            median = (sorted(data)[int(len(data) / 2)] + sorted(data)[int(len(data) / 2) + 1]) / 2
        else:
            median = sorted(data)[int(len(data) / 2)]
        return median

    # 只要results长度大于1
    def result_mean(self, relist):
        relist.sort()
        if len(relist) < 4:
            result = np.sum(np.array(relist)) / 3
        else:
            res_np = np.array(relist)[1:-1]
            length = res_np.shape[0]
            result = np.sum(res_np) / length
        return result

    def find_realbbx(self,heat,im,bbx):
        mask_heat = heat >0.1
        heat[~mask_heat] = 0
        bbx = bbx.astype(np.int32)
        cv2.polylines(im, [bbx], 1, 255)
        cv2.fillPoly(im, [bbx], 255)
        mask = im == 255
        pieceofcake = heat[mask]
        in_p = ((pieceofcake>0).sum())
        all_p = (pieceofcake.shape[0])
        if in_p == all_p:
            return True
        if in_p/all_p < 0.5:
            return True
        else:
            return False
    def result_con(self, result_list, point_list, ar=2):
        # print(sum([1,2,3,4]))
        biu = []
        pl = []
        index = []
        for i in range(0, len(result_list)):
            if len(result_list[i]) > ar:
                point_x, point_y = point_list[i]
                li = np.array(result_list[i])
                num = li.shape[0]
                w_l = li[:, 0]
                # print(w_l)
                h_l = li[:, 1]
                theta_l = li[:, 2]
                cls_l = li[:, 3]
                score_l = li[:, 4]
                score = np.sum(score_l) / num
                clse = np.sum(cls_l) / num
                w = w_l.tolist()
                h = h_l.tolist()
                theta = theta_l.tolist()
                w = self.result_mean(w)
                h = self.result_mean(h)
                theta = self.result_mean(theta)
                biu.append([point_x, point_y, w, h, theta, clse, score])
                pl.append(point_list[i])
                index.append([point_x, point_y, w, h, theta, clse, score,num])
        return biu, pl ,np.array(index)

    def get_bbx(self, pts0, scores0, result, dsets, args, down_ratio, h, w, heat, H=True):
        # pts0 = {cat: [] for cat in dsets.category}
        # scores0 = {cat: [] for cat in dsets.category}
        heat[heat<0.15] = 0
        SegMask = SegmentationMask(Heatmap=heat)
        # print(heat.shape)
        result = result.reshape(HEAT_SIZE[0] * HEAT_SIZE[1], 7)
        for i in range(0, len(result)):
            # print(result[i])
            if result[i][-1] != 0:
                im = np.zeros((heat.shape[0], heat.shape[1]), dtype="uint8")
                center_x, center_y, all_w, all_h, all_theta, all_cls, all_conf = result[i]
                # print(all_theta)
                # pts_4 = cv2.boxPoints(
                #     ((center_x, center_y), (all_w , all_h ), all_theta ))
                pts_4 = self.Decoder_reg.decoder_reg((center_x, center_y), (all_w, all_h), all_theta)
                # pts_4[:, 0] = np.clip(pts_4[:, 0],a_min=0,a_max= heat.shape[1])
                # pts_4[:, 1] = np.clip(pts_4[:, 1], a_min=0, a_max=heat.shape[0])


                cbbx = pts_4
                clse = all_cls
                # score = all_conf/(num_p+1)
                if H:

                    contour = rthombus(cbbx)
                    line = np.array([contour[1, :], contour[3,:]])
                    cv2.polylines(im, [line.astype(np.int32)], 1, 255)
                    cv2.fillPoly(im, [line.astype(np.int32)], 255)
                    # cv2.polylines(im, [contour.astype(np.int32)], 1, 255)
                    # cv2.fillPoly(im, [contour.astype(np.int32)], 255)
                    mask = im == 255
                    var_line = (heat[mask].tolist())
                    vl = [i for item in var_line for i in item]
                    # heat[mask] = 1
                    # plt.imshow(heat)
                    # plt.show()
                    le_v = int(0.3*len(vl))
                    if self.find_realbbx(heat,im,cbbx):
                        continue
                    else:
                        cv2.polylines(im, [contour.astype(np.int32)], 1, 255)
                        cv2.fillPoly(im, [contour.astype(np.int32)], 255)
                        mask = im == 255

                        num_point = (im[mask].shape[0])
                        num_sc = (np.sum(heat[mask]))


                    if num_point == 0:
                        score = 0
                    else:
                        score = num_sc / num_point
                else:
                    score = all_conf

                cbbx[:, 0] = cbbx[:, 0] * down_ratio / args.input_w * w
                cbbx[:, 1] = cbbx[:, 1] * down_ratio / args.input_h * h
                # print(score)
                if score > 0.2:
                    pts0[dsets.category[int(clse)]].append(cbbx)
                    scores0[dsets.category[int(clse)]].append(score)
                else:
                    continue

        return pts0, scores0

    def get_bbx_ltbr(self, pts0, scores0, result, dsets, args, down_ratio, h, w, heat):
        # pts0 = {cat: [] for cat in dsets.category}
        # scores0 = {cat: [] for cat in dsets.category}

        im = np.zeros((heat.shape[0], heat.shape[1]), dtype="uint8")
        pointx,pointy,l,t,r,b,theta,_,cls = result
        # print(l,t,b,r)
        # print((min(l, r) / max(l, r) * min(t, b) / max(t, b)))
        score_ratio = math.sqrt((abs(min(l, r)) / max(l, r) * abs(min(t, b) )/ max(t, b))+1e-5)
        pts_4 = self.D_ltrb.decoder_ltrb(ltrb=[l,t,r,b],angle=theta,point=(pointx,pointy))

        cbbx = pts_4
        clse = cls
        # score = all_conf/(num_p+1)

        contour = rthombus(cbbx)
        cv2.polylines(im, [contour.astype(np.int32)], 1, 255)
        cv2.fillPoly(im, [contour.astype(np.int32)], 255)
        mask = im == 255

        num_point = (im[mask].shape[0])
        num_sc = (np.sum(heat[mask]))
        if self.find_realbbx(heat,im,cbbx):
            return pts0, scores0
        else:
            if num_point == 0:
                score = 0
            else:
                score = num_sc / num_point

            score = score * score_ratio

            cbbx[:, 0] = cbbx[:, 0] * down_ratio / args.input_w * w
            cbbx[:, 1] = cbbx[:, 1] * down_ratio / args.input_h * h

            if score > 0.18:
                pts0[dsets.category[int(clse)]].append(cbbx)
                scores0[dsets.category[int(clse)]].append(score)

            return pts0, scores0


# def
def rthombus(box):
    # (4x2)
    c12 = (box[0, :] + box[1, :]) / 2
    c23 = (box[1, :] + box[2, :]) / 2
    c34 = (box[2, :] + box[3, :]) / 2
    c14 = (box[0, :] + box[3, :]) / 2
    contour = np.concatenate([c12, c23, c34, c14], axis=0).reshape(4, 2)
    return contour


# 找到一定区域内的最大值
#
def fake_maxpool(heat,input_hm, stride=3, M=False):
    input = end_zero_padding(in_array=input_hm)
    h, w = input.shape
    mask = np.zeros((h, w))
    patch_size_h = h // stride
    patch_size_w = w // stride
    for i in range(0, patch_size_h):
        for j in range(0, patch_size_w):
            patch = (input[i * (stride):(i + 1) * (stride), j * (stride):(j + 1) * (stride)])
            # print(patch)
            index = (np.where(patch == np.max(patch)))

            # print(index[0].tolist()[0]+i*(stride),index[1].tolist()[0]+j*(stride))
            x, y = index[0].tolist()[0] + i * (stride), index[1].tolist()[0] + j * (stride)
            if input[x, y] > 0:
                mask[x, y] = 1
            else:
                mask[x, y] = 0

    if M:
        mkk = mask[0:(h-1),0:(w-1)]
        plt.imshow(heat * mkk)
        plt.show()
        plt.imshow(input_hm*mask[0:(h-1),0:(w-1)])
        plt.show()
        return mask.astype(np.bool)[0:(h - 1), 0:(w - 1)]
    else:
        return input_hm * mask[0:(h - 1), 0:(w - 1)]



Decoder_reg = Decoder_REG_Targets()
from tools import Final_Treatment

def get_bbx(pts0, scores0, result, dsets, args, down_ratio, h, w, heat):
    # pts0 = {cat: [] for cat in dsets.category}
    # scores0 = {cat: [] for cat in dsets.category}
    F_T = Final_Treatment(heat)
    for i in range(0, result.shape[0]):
        # if result[i][-1] != 0:
        im = np.zeros((heat.shape[0], heat.shape[1]), dtype="uint8")
        # print(result[i])
        # center_x, center_y, all_w, all_h, all_theta, all_cls, all_conf, num_p = result[i]
        re = F_T.main(result=result[i])
        # center_x, center_y, all_w, all_h, all_theta, all_cls, all_conf = result[i]
        center_x, center_y, all_w, all_h, all_theta, all_cls, all_conf = re
        # pts_4 = cv2.boxPoints(
        #     ((center_x, center_y), (all_w / (num_p + 1), all_h / (num_p + 1)), all_theta / (num_p + 1)))

        pts_4 = Decoder_reg.decoder_reg(
            (center_x, center_y), (all_w, all_h), all_theta)
        cbbx = pts_4
        # clse = all_cls / (num_p + 1)
        clse = all_cls
        # score = all_conf/(num_p+1)
        contour = rthombus(cbbx)
        cv2.polylines(im, [contour.astype(np.int32)], 1, 255)
        cv2.fillPoly(im, [contour.astype(np.int32)], 255)
        mask = im == 255
        num_point = (im[mask].shape[0])
        num_sc = (np.sum(heat[mask]))
        if num_point == 0:
            score = 0
        else:
            score = num_sc / num_point

        cbbx[:, 0] = cbbx[:, 0] * down_ratio / args.input_w * w
        cbbx[:, 1] = cbbx[:, 1] * down_ratio / args.input_h * h

        if score > 0.25:
            pts0[dsets.category[int(clse)]].append(cbbx)
            scores0[dsets.category[int(clse)]].append(score)
        else:
            continue

    return pts0, scores0


def gaussian_label_cpu(label, num_class=18, u=0, sig=4.0):
    """
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], μ in gaussian function
        sig (float32):[1], σ in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class / 2, num_class / 2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class / 2 - label)
    return np.concatenate([y_sig[index:],
                           y_sig[:index]], axis=0)





def change_theta(angle):
    '''目前是0-90'''
    if angle >= 90:
        angle = 89.9
    angle = round(angle * 10)
    one = angle // 100
    two = (angle - one * 100) // 10
    three = angle % 10
    return one, two, three


def decoder_theta(pred_theta):
    one, two, three = pred_theta

    return one * 10 + two + three / 10



def decoder_mx(ott):
    '''用于验证label,ott在对应位置已经准备好了类似与[10,x,y],现在只要找出10中最大值对应的位置即可'''
    assert ott.shape[0] == 8
    return np.unravel_index(np.argmax(ott), ott.shape)




class SegmentationMask:
    def __init__(self,Heatmap):
        self.Heatmap = Heatmap
        self.show = Heatmap_Show()

    def get_area(self,pts4,clse):
        '''input: pts4 坐标，clse类别index'''
        '''这里要注意！！！！！！ pts4是图像坐标，切割patch的时候要转为[miny:maxy,minx:maxx]'''

        h,w,c = self.Heatmap.shape
        clse_heat = self.Heatmap[:,:,int(clse)].reshape(h,w)
        assert pts4.shape == (4,2)

        # minx, maxx, miny, maxy = np.min(pts4[:,0]),np.max(pts4[:,0]),np.min(pts4[:,1]),np.max(pts4[:,1])
        #
        # # print(minx, maxx, miny, maxy)
        # leng = 10
        # if minx > leng:
        #     minx = minx -leng
        # if miny >leng:
        #     miny = miny -leng
        # if w - maxx >leng:
        #     maxx = maxx + leng
        # if h - maxy > leng:
        #     maxy = maxy + leng

        # pts4_patch = clse_heat[int(miny):int(maxy),int(minx):int(maxx)]
        # self.show.MATPLOT_SHOW(self.Heatmap, Multiple=False)
        # res, scores, clse = self.seg(pts4_patch,vertex=(minx,miny),cls = clse)
        # self.seg(pts4_patch)
        # self.mser_seg(pts4_patch,vertex=(miny,minx))

        # return res,scores,clse


    # def seg(self,patch,vertex,cls):
    def seg(self, patch):
        res = []
        scores = []
        clse = []
        '''input: patch pred所获取的区域水平外接矩形 (minx,miny,maxx,maxy)'''
        '''vertex 左上角顶点 (minx,miny) 作用是还原坐标 获取的坐标加上 vectex就行了'''
        '''目前的问题是怎么切割，今天先用最简单的分割方式'''
        # 目前有个简单的idea 将patch*100，再用分水岭算法或者MSER算法，不过等我实现之后再说

        #首先二值化，这一部分，我们只需要一个阈值，因为patch中每个像素点的值是0-1之间
        #大于某个阈值设为1，这个应该需要经验来判断
        frame = patch.copy().reshape(patch.shape[0],patch.shape[1],1)
        # print(frame)
        img = frame.copy()
        threshold = 0.1
        mask01 = frame > threshold
        frame[mask01] = 1
        frame[~mask01] = 0
        img[mask01] = img[mask01] * 100
        img[~mask01] = 0
        frame = np.array(frame, np.uint8)
        img = np.array(img, np.uint8)

        # img_gray = cv2.Canny(img, 30,100)
        #
        # contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # cv2.drawContours(img,contours,100,(0,0,255),1)  # 画出轮廓
        # self.show.MATPLOT_SHOW(img, Multiple=False)

        # fast = cv2.FastFeatureDetector_create()
        # # 找到并绘制关键点keypoints
        # kp = fast.detect(img, None)
        # img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        # self.show.MATPLOT_SHOW(img2, Multiple=False)

        #
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)  # 画出轮廓
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # t,image_bw = cv2.threshold(image, 2, 255, cv2.THRESH_BINARY)
        #
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # image_bw = cv2.erode(image_bw, kernel, 1)
        # self.show.MATPLOT_SHOW(image_bw, Multiple=False)
        # kernel2 = np.ones((3, 3), dtype=np.uint8)
        # image_bw = cv2.dilate(image_bw, kernel2, 1)
        # self.show.MATPLOT_SHOW(image_bw, Multiple=False)
        # if
        # kernel = np.ones((5, 5), dtype=np.uint8)
        # frame = cv2.erode(frame, kernel, 1)
        self.show.MATPLOT_SHOW(frame, Multiple=False)
        # # 轮廓
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)

        for c in contours:
            # 找到边界坐标
            # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            im = np.zeros((patch.shape[0], patch.shape[1]), dtype="uint8")
            # 找面积最小的矩形
            rect = cv2.minAreaRect(c)
            # 得到最小矩形的坐标
            cbbx = cv2.boxPoints(rect)
            # contour = rthombus(cbbx)
            # cv2.polylines(im, [contour.astype(np.int32)], 1, 255)
            # cv2.fillPoly(im, [contour.astype(np.int32)], 255)
            # mask = im == 255
            # num_point = (im[mask].shape[0])
            # num_sc = (np.sum(patch[mask]))
            # if num_point == 0:
            #     score = 0
            # else:
            #     score = num_sc / num_point
            # cbbx[:,0] = cbbx[:,0] + vertex[0]
            # cbbx[:, 1] = cbbx[:, 1] + vertex[1]
            # if score > 0.4:
            #     res.append(cbbx)
            #     scores.append(score)
            #     clse.append(cls)

            # print(box)
            # # 标准化坐标到整数
            box = np.int0(cbbx)
            # # 画出边b
            cv2.drawContours(image, [box], 0, (0, 0, 255), 1)




        self.show.MATPLOT_SHOW(image,Multiple=False)
        # return res,scores,clse


    def mser_seg(self,patch,vertex):
        frame = patch.copy().reshape(patch.shape[0], patch.shape[1], 1)
        threshold = 0.1
        mask01 = frame > threshold
        frame[~mask01] = 0
        gray = frame*200
        gray = np.array(gray, np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        mser = cv2.MSER_create()  # 得到mser算法对象
        regions, _ = mser.detectRegions(gray)  # 获取文本区域
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(image, hulls, 1, (255, 0, 0))
        cv2.namedWindow("img", 0)
        cv2.resizeWindow("img", 800, 640)  # 限定显示图像的大小
        cv2.imshow('img', image)
        cv2.waitKey(0)




class center_pooling:
    def __init__(self,ratio=0.9):
        self.ratio = ratio

    def in_roundness(self,point,center,radius):
        '''input: point坐标点 center 圆心，指代图中index最大的坐标点的位置，radius代表园的半径，使用最短边'''
        '''output: bool'''
        point_x,point_y = point
        center_x,center_y = center
        P_R = (point_x-center_x)**2 + (point_y-center_y)**2
        R_R = (radius/2)**2
        if P_R < R_R or P_R == R_R:
            return True
        else:
            return False

    def in_square(self,point,center,los):
        '''input: point坐标点 center 圆心，指代图中index最大的坐标点的位置，los length of side，使用最短边'''
        cx,cy = center
        px,py = point
        minx,miny,maxx,maxy = cx-los,cy-los,cx+los,cy+los
        if px < minx or py < miny or px > maxx or py > maxy:
            return False
        else:
            return True


    def c_pooling(self,result,terminal):
        '''找出区域范围内最大的两个点'''
        '''这个区域范围的定义是最短的边围成的区域 '''
        '''result 已经排好序的 '''
        # print(terminal)
        if len(result) == 0:
            return terminal
        assert result.shape[1] == 8
        num = result.shape[0]
        terminal.append(result[0].tolist())

        if num == 1:
            return terminal

        max_p,los = result[0,0:2],result[0,3]
        print(los)
        points = result[1:,0:2]
        re = result[1:]
        t_points = points - max_p
        mask_p = (np.abs(t_points[:,0])>los) &(np.abs(t_points[:,1])>los)
        if len(points[~mask_p]) > 1:
            second_max_p = points[~mask_p][0]
            sec_index = np.argwhere((result[:, 0] == second_max_p[0]) & (result[:, 1] == second_max_p[1]))[0][0]
            terminal.append(result[sec_index].tolist())


        return self.c_pooling(re[mask_p],terminal)


