import os

import cv2
import numpy as np
# from pooling import *
import math
import matplotlib.pyplot as plt
from my_augments import *

INPUT_SIZE = (720,1280)
HEAT_SIZE = (INPUT_SIZE[0] //4,INPUT_SIZE[1] //4)
'''查看图像'''
class Heatmap_Show:
    def CV2_SHOW(self,heatmap,Multiple=False,save=False):
        if Multiple:
            print('< class >:',type(heatmap))
            assert isinstance(heatmap, list)
            assert len(heatmap) == 3
            heats = np.hstack(heatmap[0],heatmap[1],heatmap[2])#横向拼接
            # heats = np.vstack(heatmap[0], heatmap[1], heatmap[2])  # 纵向拼接 组合使用

        else:
            heats = heatmap

        cv2.namedWindow('Heat-Map')
        cv2.imshow('Heat-Map',heats)
        cv2.waitKey(0)

    def MATPLOT_SHOW(self,heatmap,Multiple=False):
        if Multiple:
            print('< class >:',type(heatmap))
            assert isinstance(heatmap, list)
            assert len(heatmap) == 3
            plt.figure()
            plt.subplot(131)  #1 代表位置 3 代表图的个数 最后一位代表index
            plt.imshow(heatmap[0])
            plt.subplot(132)
            plt.imshow(heatmap[1])
            plt.subplot(133)
            plt.imshow(heatmap[2])
            plt.show()
        else:
            plt.imshow(heatmap)
            plt.show()

'''建立高斯映射'''
class Ellipse_Gaussian:
    def __init__(self):
        self.SHOW = Heatmap_Show()
    def coordinates(self, size):
       '''将宽高形如（10，10）的尺寸计算的到对应的图像系坐标'''
       '''output: (w*h,2)'''
       size_x, size_y = size
       shifts_x = np.arange(0, size_x, 1, dtype=np.float32)
       shifts_y = np.arange(0, size_y, 1, dtype=np.float32)
       shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
       shift_x = np.reshape(shift_x, [-1])
       shift_y = np.reshape(shift_y, [-1])
       coords = np.stack([shift_x, shift_y], -1)
       return coords

    def fragment_center(self,size):
        '''返回切块中心坐标'''
        return ((size[0]-1)//2+1,(size[1]-1)//2+1)
        # return (size[0], size[1])

    def rectangular_coordinates(self,size):
        '''切块坐标系转直角坐标系'''
        '''此处x坐标应为-x,不过不影响高斯分布'''
        '''中心点选取：最长边h形成的切块，最好为单数，则中心点为(h-1)/2+1'''
        return self.fragment_center(size) - self.coordinates(size)

    def normal_gaussian_2D(self,size,show=False):
        fragment = self.coordinates(size)
        sigma = (size[0]-1)//2
        center_x,center_y = self.fragment_center(size)
        gaussian = np.exp(-((fragment[:,0]-center_x)**2+(fragment[:,1]-center_y)**2)/(2*sigma**2))
        gaussian = gaussian.reshape(size[0],size[1])
        if show:
            self.SHOW.MATPLOT_SHOW(gaussian)

        return gaussian
    def ellipse_gaussian_2D(self,size,angle,wh,show=False):
        '''高斯分布与协方差，上一个版本用的旋转像素点，效果不怎么好，更新之后舒服了'''
        '''input: size (?,?)代表块的尺寸，为正方形，由最长边给出，将最长边变换为奇数，这样能够使得中心点在最中心'''
        '''wh: h代表数值方向，w为水平方向'''
        '''angle: 角度范围[0,180),180与0是一样的 其中若w>h 90一下顺时针，90-180逆时针,若w<h 则相反'''
        fragment = self.coordinates(size)
        center_x, center_y = self.fragment_center(size)
        w,h = wh
        theta = math.radians(angle) #弧度
        matrix_left = np.array([math.cos(theta),-math.sin(theta),math.sin(theta),math.cos(theta)]).reshape(2,2)#逆时针
        matrix_middle = np.array([w/2,0,0,h/16]).reshape(2,2)
        matrix_right = np.array([math.cos(theta), math.sin(theta), -math.sin(theta), math.cos(theta)]).reshape(2, 2)#or matrix_left.T 顺时针

        matrix_covariance = np.matmul(np.dot(matrix_left,matrix_middle),matrix_right)

        formula = np.stack([fragment[:,0]-center_x, fragment[:,1]-center_y], -1)
        formula_left = formula
        formula_right = formula.T

        formula_ = [np.matmul(np.matmul(formula_left[i,:].reshape(1,2),matrix_covariance),formula_right[:,i].reshape(2,1)) for i in range(formula_left.shape[0])]
        formula_ = np.array(formula_)
        ellipse_gaussian = np.exp(-formula_[:,0]/(2*(w/4)**2*(h/8)**2+1e-5))
        ellipse_gaussian = ellipse_gaussian.reshape(size[0],size[0])
        if show:
            self.SHOW.MATPLOT_SHOW(ellipse_gaussian)


        return ellipse_gaussian



''''''
class Encoder_LTBR:
    def one_line(self, center, top):
        '''计算两点连成的直线坐标'''
        cx, cy = center
        tx, ty = top
        A = ty - cy
        B = cx - tx
        C = tx * cy - cx * ty
        return [A, B, C]

    def distance(self, point, p1, p2):
        A, B, C = self.one_line(p1, p2)
        x, y = point[0], point[1]
        dis = np.abs((A * x + B * y + C) / math.sqrt(A * A + B * B))
        return dis

    def ltrb(self,bbx,point):
        # bbx[:,1] = -bbx[:,1]
        l = self.distance(point, bbx[0, :], bbx[3, :])
        t = self.distance(point,bbx[0,:],bbx[1,:])
        r = self.distance(point, bbx[1, :], bbx[2, :])
        b = self.distance(point, bbx[3, :], bbx[2, :])
        return np.array([l,t,r,b])


    def decoder_ltrb(self,ltrb,angle,point):
        l,t,r,b = ltrb
        cen_x, cen_y = point

        if angle > 90:
            angle = angle - 180
        else:
            angle = angle
        theta = math.radians(angle)
        # theta = 90 - theta
        # print(theta)
        bbx_x_asix = [[-l, t], [r, t ], [r , -b], [-l , -b ]]
        matrix_left = np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,
                                                                                                              2)  # 逆时针
        img_bbx = []
        for coor in bbx_x_asix:
            img_coor = np.matmul(matrix_left, np.array(coor).reshape(2, 1))
            img_coor_x, img_coor_y = cen_x + img_coor[0], cen_y - img_coor[1]
            img_bbx.append([img_coor_x, img_coor_y])
        bbx = np.array(img_bbx).reshape(4, 2)
        return bbx
'''用于构建全卷积分类特征图，可以为高斯热图或者分割图'''
class Encoder_CLS_target:
    def __init__(self, bbxes):
        super(Encoder_CLS_target).__init__()
        self.bbxes = bbxes
        self.center = np.array([np.sum(bbxes[:,0]/4),np.sum(bbxes[:,1]/4)])
        self.EG = Ellipse_Gaussian()
        self.DC = Decoder_REG_Targets()
    def odd(self,length):
        if length %2 == 0:
            length = int(length)
            return (length+1)
        else:
            length = int(length)
            return length

    def _box(self):
        '''检测盒子的尺寸是(8,) or (4,2)'''
        '''output: (4,2)'''
        # box = self.bbxes.astype(np.int32)
        box = self.bbxes
        if box.shape[0] > 4:
            box = np.array(box)
            bbox = box.reshape(4, 2)
        else:
            bbox = box
        return bbox

    def box_zooming(self,results,ratio=0.8):
        bbox_w, bbox_h = results['wh']
        theta = results['theta']
        cen_x, cen_y = results['center']
        w ,h = ratio*bbox_w,ratio*bbox_h*0.5
        box_z = self.DC.decoder_reg((cen_x,cen_y),(w,h),theta)

        return box_z

    def patch(self):
        input_size_y,input_size_x = INPUT_SIZE[0],INPUT_SIZE[1]
        '''从bbox种剥离出来，最小外接水平矩形，用于贴在CLS特征图上'''
        box = self._box()
        minbbx, minbby = np.min(box[..., 0]), np.min(box[..., 1])
        maxbbx, maxbby = np.max(box[..., 0]), np.max(box[..., 1])
        # mm = np.array([minbbx, minbby, maxbbx, maxbby])
        clip_y = np.array([minbby,maxbby])
        clip_y = np.asarray(np.clip(clip_y, a_min=1., a_max=(input_size_y // 4) - 1.), np.int32)
        clip_x = np.array([minbbx, maxbbx])
        clip_x = np.asarray(np.clip( clip_x , a_min=1., a_max=(input_size_x // 4) - 1.), np.int32)
        minbx, maxbx = clip_x.tolist()
        minby, maxby = clip_y.tolist()
        size = [maxby - minby, maxbx - minbx]#为什么是（y,x）的形式，cls特征图上的坐标与图像坐标正好相反
        size = list(map(int, size))
        center = self.center.astype(np.int32)
        fake_center = center - np.array([minbx, minby], dtype=np.int32)

        return size, fake_center, [minbx, minby, maxbx, maxby], np.array([minbx, minby], dtype=np.int32)

    def mask_patchbox_onepoint(self, box, point):
        '''利用向量叉乘的方式，得到patch中物体所在区域的mask'''
        '''input: box:目标的包围盒，这里的包围盒已经转换为patch中的包围盒 point：patch中各个点的坐标'''
        P1, P2, P3, P4 = box
        P21,P43,P32,P14 = P2 - P1,P4 - P3,P3 - P2,P1 - P4
        v1,v2,v3,v4 = point - P1,point - P2,point - P3,point - P4
        mask = (np.cross(P21, v1) * np.cross(P43, v3) >= 0) & (np.cross(P32, v2) * np.cross(P14, v4) >= 0)
        return mask
    def mask_patchbox_all(self,minxy,size,rth=True):

        corner = self._box() - minxy
        coords = self.EG.coordinates(size)
        corner = np.concatenate((corner[:, 1].reshape(4, 1), corner[:, 0].reshape(4, 1)), axis=-1)
        P1,P2,P3,P4 = corner[0,:],corner[1,:],corner[2,:],corner[3,:]

        C12, C23, C34, C14 = (P1 + P2) / 2 , (P2 + P3) / 2 , (P3 + P4) / 2, (P1 + P4) / 2


        if rth:
            msk = np.array([self.mask_patchbox_onepoint([C12, C23, C34, C14], i) for i in coords])  # 菱形叉乘

        else:
            msk = np.array([self.mask_patchbox_onepoint([P1, P2, P3, P4], i) for i in coords])

        msk = (msk.reshape((size[0], size[1])))



        return msk
    def mask_patchbox_zooming(self,minxy,size,box_z,rth=False):

        corner = box_z - minxy
        coords = self.EG.coordinates(size)
        corner = np.concatenate((corner[:, 1].reshape(4, 1), corner[:, 0].reshape(4, 1)), axis=-1)
        P1,P2,P3,P4 = corner[0,:],corner[1,:],corner[2,:],corner[3,:]

        C12, C23, C34, C14 = (P1 + P2) / 2 , (P2 + P3) / 2 , (P3 + P4) / 2, (P1 + P4) / 2

        if rth:
            msk = np.array([self.mask_patchbox_onepoint([C12, C23, C34, C14], i) for i in coords])  # 菱形叉乘

        else:
            msk = np.array([self.mask_patchbox_onepoint([P1, P2, P3, P4], i) for i in coords])

        msk = (msk.reshape((size[0], size[1])))


        return msk
    def patch_on_heatmap(self,Heatmap,results,G=False):
        mask_heatmap = Heatmap > 0
        before_heat = Heatmap[mask_heatmap]
        '''只需要计算中心点对左上角以及右上角的偏移来确定area'''
        cen_x, cen_y = results['center']
        assert int(self.center[0]) == int(cen_x) and int(self.center[1]) == int(cen_y)
        bbox_w, bbox_h = results['wh']
        theta = results['theta']
        # flos = results['offsets']
        AP_size,AP_center,Rect_box,Left_point = self.patch()
        minbx, minby, maxbx, maxby = Rect_box #目标框在heatmap上的位置 np.int32
        bbx_cx, bbx_cy = self.center.astype(np.int32)
        left_ox ,left_oy = minbx - bbx_cx ,minby - bbx_cy
        right_ox, right_oy = maxbx - bbx_cx, maxby - bbx_cy
        box_z = self.box_zooming(results)
        Area_mask = self.mask_patchbox_all(Rect_box[0:2],size=AP_size)
        Area_mask_narrow = self.mask_patchbox_zooming(Rect_box[0:2], box_z=box_z, size=AP_size)
        GS_size = (self.odd(round(bbox_w)+30),self.odd(round(bbox_w))+30)
        point_area = 100 # 采样问题 限定采样点为100个小于100就是 bbx所有区域
        area = bbox_w * bbox_h
        # if Area_mask.sum():
        #     if Area_mask.sum() < point_area:
        #         # Area_mask = self.mask_patchbox_all(Rect_box[0:2],size=AP_size,rth=False)
        #         Heatmap[minby:maxby, minbx:maxbx][Area_mask] = 1
        #     else:
        #         Heatmap[minby:maxby, minbx:maxbx][Area_mask_narrow] = 1
        '''llll'''
                # GS_size = (self.odd(round(bbox_w) + 20), self.odd(round(bbox_w)) + 20)
                # Ell_Gaussian = self.EG.ellipse_gaussian_2D(GS_size, theta, (round(bbox_w), round(bbox_h)))
                # EG_center_y, EG_center_x = self.EG.fragment_center(GS_size)  # 注意 x,y的位置
                # # EG_minbx, EG_minby, EG_maxbx, EG_maxby = left_ox + EG_center_x, left_oy + EG_center_y, right_ox + EG_center_x, right_oy + EG_center_y
                # EG_minbx, EG_minby = left_ox + EG_center_x, left_oy + EG_center_y
                # EG_maxbx, EG_maxby = EG_minbx +(maxbx-minbx) ,EG_minby +(maxby-minby)
                #
                # EG_mask = Ell_Gaussian > 0.1
                # Ell_Gaussian[~EG_mask] = 0
                # mask_a = Ell_Gaussian > 0.8
                # if mask_a.sum()>100:
                #     mask_a = Ell_Gaussian > 0.85
                # else:
                #     mask_a = Ell_Gaussian > 0.75
                # Ell_Gaussian[mask_a] = 1
                # Ell_Gaussian[~mask_a] = 0
                # # print(mask_a.sum())
                # # print(minby,maxby, minbx,maxbx)
                # # print(EG_minby,EG_maxby, EG_minbx,EG_maxbx)
                # # print(Heatmap.shape)
                # # print(Ell_Gaussian.shape)
                # Heatmap[minby:maxby, minbx:maxbx] = Ell_Gaussian[EG_minby:EG_maxby, EG_minbx:EG_maxbx]


        # Ell_Gaussian = self.EG.ellipse_gaussian_2D(GS_size,theta,(round(bbox_w),round(bbox_h)))
        # EG_center_y,EG_center_x = self.EG.fragment_center(GS_size) #注意 x,y的位置
        # EG_minbx , EG_minby ,EG_maxbx, EG_maxby = left_ox + EG_center_x ,left_oy + EG_center_y, right_ox + EG_center_x, right_oy + EG_center_y

        if G:
            Ell_Gaussian = self.EG.ellipse_gaussian_2D(GS_size, theta, (round(bbox_w), round(bbox_h)))
            EG_center_y, EG_center_x = self.EG.fragment_center(GS_size)  # 注意 x,y的位置
            EG_minbx, EG_minby, EG_maxbx, EG_maxby = left_ox + EG_center_x, left_oy + EG_center_y, right_ox + EG_center_x, right_oy + EG_center_y
            EG_mask = Ell_Gaussian > 0.43

            if EG_mask.sum():
                mask_a = Ell_Gaussian > 0.65
                Ell_Gaussian[~EG_mask] = 0
                Ell_Gaussian[EG_mask] = 1

                # if Heatmap[minby:maxby, minbx:maxbx].shape != Ell_Gaussian[EG_minby:EG_maxby, EG_minbx:EG_maxbx].shape:
                # print(Heatmap[minby:maxby, minbx:maxbx].shape ,Ell_Gaussian[EG_minby:EG_maxby, EG_minbx:EG_maxbx].shape)
                if(EG_maxbx-EG_minbx)!=(maxbx-minbx):
                    EG_maxbx = EG_maxbx +1
                Heatmap[minby:maxby, minbx:maxbx] = Ell_Gaussian[EG_minby:EG_maxby, EG_minbx:EG_maxbx]
        else:
            if Area_mask.sum():
                # Heatmap[minby:maxby, minbx:maxbx][Area_mask] = 1
                Heatmap[minby:maxby, minbx:maxbx][Area_mask_narrow] =1


        if mask_heatmap.sum():
            Heatmap[mask_heatmap] = before_heat


        # mask_some = (Heatmap > 0) & (Heatmap<0.5)
        # Heatmap[mask_some] =Heatmap[mask_some]+0.5
        return Heatmap
'''用于在回归特征图上打标签'''
class Encoder_REG_Targets:
    def __init__(self):
        self.EG = Ellipse_Gaussian()
        self.DR = Decoder_REG_Targets()
        self.E_ltrb = Encoder_LTBR()

    def change_theta_(self,theta):
        one = theta // 10  # 十位以前的数
        two = theta - one*10
        return one, two

    def center_offset(self,point,center):
        x , y = point
        return center[0] - x, center[1] - y
    def encoder_label(self,results,reg_fm,theta_fm_reg,theta_fm_cls,mask,hm11,cls=True):
        int_w,int_h = HEAT_SIZE
        cen_x, cen_y = results['center']
        bbox_w, bbox_h = results['wh']
        theta = results['theta']
        flos = results['offsets']
        # print(theta)
        # print(self.change_theta_(theta))
        # bbx = self.DR.decoder_reg(center=(cen_x+flos[0],cen_y+flos[1]),wh=(bbox_w,bbox_h),angle=theta)
        ct = np.asarray([cen_x, cen_y], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        # print(ct_int)
        coordss = self.EG.coordinates(size=(int_w,int_h))
        coordss = coordss.reshape((int_w,int_h, 2))
        for i in range(0, coordss[mask].shape[0]):
           x, y = (coordss[mask][i][0], coordss[mask][i][1])
           Tx_off,Ty_off = self.center_offset((y,x),(ct_int[0], ct_int[1]))
           # print(Tx_off,Ty_off)
           # ltrb = self.E_ltrb.ltrb(bbx, (y, x))
           # reg_fm[int(x), int(y), 0:] = ltrb
           reg_fm[int(x), int(y), 0:2] = np.array([bbox_w, bbox_h])
           reg_fm[int(x), int(y), 2:4] = np.array([Tx_off, Ty_off])
           reg_fm[int(x), int(y), 4:] = flos
           hm11[int(x), int(y)] = 1
           if cls:
               one, two = self.change_theta_(theta)
               theta_fm_cls[int(one), int(x), int(y)] = 1.0
               # for i in range(0,18):
               #     if int(i) == int(one):
               #         theta_fm_cls[int(i), int(x), int(y)] = 0.9
               #     else:
               #         theta_fm_cls[int(i), int(x), int(y)] = (1-0.9)/18
               theta_fm_reg[int(x), int(y), :] = two
           else:
               theta_fm_reg[int(x), int(y), :] = theta

           # l,t,r,b = self.E_ltrb.ltrb(bbx,(y,x))
           # if y == ct_int[0] and x ==ct_int[1]:
           #     print(l,t,r,b)
           # if (math.sqrt(min(l,r)/max(l,r)*min(t,b)/max(t,b)))>0.9:
           #     print(l,t,r,b)
           #     print(y,x)
           #     print(ct_int)
           # ebbx = self.E_ltrb.decoder_ltrb(ltrb,theta,(y,x))
           # print(bbx)
           # print(ebbx)
           # print(y,x)
           # im_one = np.zeros((HEAT_SIZE[0], HEAT_SIZE[1]), dtype="uint8")
           # im_two = np.zeros((HEAT_SIZE[0], HEAT_SIZE[1]), dtype="uint8")
           # cv2.polylines(im_one, [bbx.astype(np.int32)], 1, 1)
           # cv2.fillPoly(im_one, [ bbx.astype(np.int32)], 1)
           # cv2.polylines(im_two, [ebbx.astype(np.int32)], 1, 1)
           # cv2.fillPoly(im_two, [ ebbx.astype(np.int32)], 1)
           # plt.imshow(im_one)
           # plt.show()
           # plt.imshow(im_two)
           # plt.show()
        return reg_fm,theta_fm_reg,theta_fm_cls,hm11




'''用于将回归特征图上的label转换为真实的bbx并查看'''
class Decoder_REG_Targets:
    def decoder_cf(self):
        return
    def decoder_reg(self,center,wh,angle,nine=False):
        '''part1.中心点，尺寸，角度[0-180)encoder的时候是根据最长边与x轴正向的夹角'''
        '''part2.中心点，尺寸，角度[0-90)encoder的时候是根据cv2.minAreaRect获取的'''
        if nine:
            cen_x,cen_y = center
            bbox_w, bbox_h = wh
            theta = angle
            bbx = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))
        else:
            h, w = wh
            cen_x, cen_y = center

            if angle > 90:
                angle = angle - 180
            else:
                angle = angle
            theta = math.radians(angle)
            # theta = 90 - theta
            # print(theta)
            bbx_x_asix = [[-h/2,w/2],[h/2,w/2],[h/2,-w/2],[-h/2,-w/2]]
            matrix_left = np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,2)  # 逆时针
            img_bbx = []
            for coor in bbx_x_asix:
                img_coor = np.matmul(matrix_left,np.array(coor).reshape(2,1))
                img_coor_x ,img_coor_y = cen_x + img_coor[0] ,cen_y - img_coor[1]
                img_bbx.append([img_coor_x ,img_coor_y])
            bbx = np.array(img_bbx).reshape(4,2)
        return bbx


'''根据对角度进行处理，或将角度进行分类或者回归处理'''
class Theta_Switch:
    def __init__(self,bbx):
        self.bbx = bbx
        # self.cv_theta = cv_theta

    def distance(self,P1,P2):
        '''计算两点间距离'''
        dis = np.sqrt(np.square(P1[0] - P2[0]) + np.square(P1[1] - P2[1]))
        return dis
    def to_xasix(self,P1,P2):
        x1,y1 = P1
        x2,y2 = P2
        if y1 < y2:
            vec = (x1-x2,y2-y1)
        elif y1 == y2:
            if x1>x2:
                vec = (x1-x2,0)
            else:
                vec = (x2-x1,0)
        else:
            vec = (x2-x1,y1-y2)
        return vec
    def azimuthAngle(self,x1, y1, x2=0, y2=0):
        angle = 0.0
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        result = (angle * 180 / math.pi)
        if result > 180:
            return result - 180
        elif result == 180:
            return 179.899
        else:
            return result

    def Theta(self,nine=False):
        box = self.bbx
        results = {}
        if nine:
            rect = cv2.minAreaRect(box)
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            flos = ct - ct_int
            results['center'] = [cen_x, cen_y]
            results['wh'] = [bbox_w, bbox_h]
            results['theta'] = theta
            results['offsets'] = flos
        else:
            c12,c23,c34,c14 = (box[0, :] + box[1, :]) / 2, (box[1, :] + box[2, :]) / 2,(box[2, :] + box[3, :]) / 2,(box[0, :] + box[3, :]) / 2
            cen_x, cen_y = np.sum(box[:,0])/4 ,np.sum(box[:,1])/4
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            flos = ct - ct_int
            if self.distance(c12, c34) > self.distance(c23, c14):
                theta = self.vec_to_theta(c12, c34)
                bbox_w, bbox_h = self.distance(c12, c34), self.distance(c23, c14)  # w长边
                # return theta
            elif self.distance(c12, c34) == self.distance(c23, c14):
                bbox_w, bbox_h = self.distance(c12, c34),self.distance(c23, c14)
                theta1 = self.vec_to_theta(c12, c34)
                theta2 = self.vec_to_theta(c23, c14)
                theta = min(theta1,theta2)
            else:
                theta = self.vec_to_theta(c23,c14)
                bbox_w, bbox_h = self.distance(c23, c14), self.distance(c12, c34)#w长边

            results['center'] = [cen_x, cen_y]
            results['wh'] = [bbox_w, bbox_h]
            results['theta'] = theta
            results['offsets'] = flos
        return results

    def vec_to_theta(self,point1, point2):
        px1, py1 = point1
        px2, py2 = point2
        x2, y2 = 2, 0
        if px1 > px2:
            vx1, vy1 = px1 - px2, py1 - py2
        else:
            vx1, vy1 = px2 - px1, py2 - py1
        x1, y1 = vx1, -vy1

        cosa = (x1 * x2 + y1 * y2) / (math.sqrt(x1 * x1 + y1 * y1) * math.sqrt(x2 * x2 + y2 * y2))
        cosa = np.clip(np.array([cosa]),-1.0,1.0)[0]
        theta = math.degrees(math.acos(cosa))
        # print(theta)
        if y1 >= 0:
            theta = theta
        else:
            theta = 180 - theta
        if theta >= 179:
            theta = 179.59
        theta = int(theta*100) / 100
        # print(theta)
        return theta





'''对最终的结果进行解码'''
'''增加了对中心点位置偏移的回归，最终的结果蛮好的'''
'''尺寸预测不达标'''
# class Decoder_Detection_Results:

'''数据增强，原来写的代码可能优点问题'''
class IM_AUG:
    def quad_2_rbox(self,quads, mode='xyxya'):
        # http://fromwiz.com/share/s/34GeEW1RFx7x2iIM0z1ZXVvc2yLl5t2fTkEg2ZVhJR2n50xg
        if len(quads.shape) == 1:
            quads = quads[np.newaxis, :]
        rboxes = np.zeros((quads.shape[0], 5), dtype=np.float32)
        for i, quad in enumerate(quads):
            rbox = cv2.minAreaRect(quad.reshape([4, 2]))
            x, y, w, h, t = rbox[0][0], rbox[0][1], rbox[1][0], rbox[1][1], rbox[2]
            if np.abs(t) < 45.0:
                rboxes[i, :] = np.array([x, y, w, h, t])
            elif np.abs(t) > 45.0:
                rboxes[i, :] = np.array([x, y, h, w, 90.0 + t])
            else:
                if w > h:
                    rboxes[i, :] = np.array([x, y, w, h, -45.0])
                else:
                    rboxes[i, :] = np.array([x, y, h, w, 45])
        # (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        if mode == 'xyxya':
            rboxes[:, 0:2] = rboxes[:, 0:2] - rboxes[:, 2:4] * 0.5
            rboxes[:, 2:4] = rboxes[:, 0:2] + rboxes[:, 2:4]
        rboxes[:, 0:4] = rboxes[:, 0:4].astype(np.int32)
        return rboxes

    def mask_valid_boxes(self,boxes, return_mask=False):
        """
        :param boxes: (cx, cy, w, h,*_)
        :return: mask
        """
        w = boxes[:, 2]
        h = boxes[:, 3]
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        mask = (w > 2) & (h > 2) & (ar < 30)
        if return_mask:
            return mask
        else:
            return boxes[mask]

    def im_augment(self,image, annotation, augment=True):
        ann = {}
        obj = annotation['pts'].shape[0]
        bbx = annotation['pts']
        cls = annotation['cat']
        dif = annotation['dif']

        if augment:
            transform = Augment([HSV(0.5, 0.5, p=0.5),#Augment中多有的p代表概率 0.5 假设100个样本 会有44-50个样本进行改变
                                 HorizontalFlip(p=0.5),
                                 VerticalFlip(p=0.5),
                                 Affine(degree=30, translate=0.1, scale=0.1, p=0.7),
                                 Noise(0.01, p=0.3),
                                 Blur(1.3, p=0.5),
                                 ], box_mode='xyxyxyxy')
            im, bboxes = transform(image, bbx)
            mask = self.mask_valid_boxes(self.quad_2_rbox(bboxes, 'xywha'), return_mask=True)
            bboxes = bboxes.reshape((obj, 4, 2))
            bbox = bboxes[mask]
            cls = cls[mask]
            dif = dif[mask]
        else:
            im, bbox = image, bbx
        # for i ,bbx in enumerate(bbox):
        #     bx = np.array(bbx).reshape(4, 2)
        #     cv2.polylines(im, [bx.astype(np.int32)], 1, 255)
        # HS = Heatmap_Show()
        # HS.CV2_SHOW(im, Multiple=False)

        ann['pts'] = bbox
        ann['cat'] = cls
        ann['dif'] = dif
        return im, ann

    def prepare_data(self,im,ann_in,size=(INPUT_SIZE[1],INPUT_SIZE[0]),down_ratio=4):
        ann_out = {}
        labels = ann_in['pts']
        cls = ann_in['cat']
        # dif = ann_out['dif']
        # 输入的bbx为(numobj,4,2) 或者(numobj, 8,) 测试的时候不用
        h, w = im.shape[:2]
        img = cv2.resize(im, size)
        scale_h, scale_w = size[1] / h, size[0] / w
        # for i, label in
        # labels[:, 0] = labels[:, 0] * scale_h
        # labels[:, 1] = labels[:, 1] * scale_w
        # for i ,bbx in enumerate(labels):
        #     bx = np.array(bbx).reshape(4, 2)
        #     cv2.polylines(img, [bx.astype(np.int32)], 1, 255)
        # HS = Heatmap_Show()
        # HS.CV2_SHOW(img, Multiple=False)
        cv_label, out_cat = [], []
        imgg = cv2.resize(img, (size[0]//down_ratio,size[1]//down_ratio))
        # print(imgg.shape)
        # print(labels)
        for i, quad in enumerate(labels):
            quad[:, 0] = quad[:, 0] * scale_w
            quad[:, 1] = quad[:, 1] * scale_h
            quads = quad / down_ratio
            # print(quads)
            # bx = np.array(quads).reshape(4, 2)
            # cv2.polylines(imgg, [bx.astype(np.int32)], 1, 255)

            # rect = cv2.minAreaRect(quads.reshape([4, 2]))
            # pts_4 = cv2.boxPoints(((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]),  rect[2]))
            # print(pts_4)
            # cv_label.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
            cv_label.append(quads)
            out_cat.append(cls[i])
        # HS = Heatmap_Show()
        # HS.CV2_SHOW(imgg, Multiple=False)
        ann_out['rect'] = np.array(cv_label,dtype=np.float32)
        ann_out['cat'] = np.asarray(out_cat, np.uint8)
        # drew_box(img, label)
        # drew_box(cv2.resize(img, (size[0] // 4, size[1] // 4)), label / 4)
        # print(ann_out['rect'].shape)
        return img,ann_out
def drew_box(im,bbx):
    bbx = np.array(bbx).reshape(4, 2)
    cv2.polylines(im, [bbx.astype(np.int32)], 1, 255)
    HS = Heatmap_Show()
    HS.CV2_SHOW(im,Multiple=False)



def watershed_algorithm(image):
    # 边缘保留滤波EPF  去噪
    blur = cv2.pyrMeanShiftFiltering(image,sp=10,sr=100)
    # 转成灰度图像
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # 得到二值图像   自适应阈值
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', binary)

    # 形态学操作   获取结构元素  开操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # 确定区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # cv.imshow('mor-opt', sure_bg)

    # 距离变换
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    # cv.imshow('distance-', dist_out * 50)
    ret, surface = cv2.threshold(dist_out, dist_out.max() * 0.6, 255, cv2.THRESH_BINARY)
    # cv.imshow('surface-markers', surface)

    surface_fg = np.uint8(surface)    # 转成8位整型
    unkonown = cv2.subtract(sure_bg, surface_fg)        # 找到位置区域
    # Marker labelling
    ret, markers = cv2.connectedComponents(surface_fg)  # 连通区域

    # 分水岭变换
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unkonown == 255] = 0
    # 实施分水岭算法了。标签图像将会被修改，边界区域的标记将变为 -1
    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]      # 被标记的区域   设为红色
    cv2.imshow('result', image)




'''功能测试'''

import torch.nn.functional as F

'''这个用于将圆圈内的点，筛选出来'''


class screen:
    def __init__(self, detection_result):
        self.center_result = detection_result.cpu().numpy()
        # print(self.center_result)

    def in_roundness(self, point,nump, center,numc, radius):
        '''input: point坐标点 center 圆心，指代图中index最大的坐标点的位置，radius代表园的半径，使用最短边'''
        '''output: bool'''
        point_x, point_y = point
        center_x, center_y = center
        P_R = (point_x - center_x) ** 2 + (point_y - center_y) ** 2
        R_R = (radius / 2) ** 2
        if P_R <= R_R and nump<numc:
            return False
        else:
            return True

    def surrounding_SUM(self, coor, zer_t_n):
        '''已知mask求索引？目前的解决方案就是，坐标图，zer图，根据坐标求'''
        '''目前则是给定一个坐标，对坐标八邻域范围内的值求和'''
        '''[105.,  67.]'''

        parm_l1 = [[-1, -1], [1, 1], [0, 1], [0, -1], [1, -1], [-1, 1], [1, 0], [-1, 0]]
        cx, cy = coor
        cx, cy = int(cx), int(cy)
        value_center = zer_t_n[cy, cx]
        for eig in parm_l1:
            off_x, off_y = eig[0] + cx, eig[1] + cy
            value_center = value_center + zer_t_n[off_y, off_x]

        #
        return value_center
    def main(self,all_center,zer_t_n):
        all_center = all_center.cpu().numpy()
        v_c = []
        for center in all_center:
            v_c.append(self.surrounding_SUM(center,zer_t_n))

        v_c = np.array(v_c).reshape(len(v_c),1)
        return  self.decoupling(np.concatenate([self.center_result,v_c],axis=-1))

    def decoupling(self,result):
        '''input-shape:n,8'''
        mask_l = np.ones((result.shape[0],1),dtype=bool)
        # print(mask_l)
        for re in result:
            cent = re[0:2]
            h = re[3]
            value = re[-1]
            # print(cent,h,value)
            m = []
            for o_re in  result:
                o_cent = o_re[0:2]
                o_h = o_re[3]
                o_value = o_re[-1]
                mask = self.in_roundness(o_cent,o_value,cent,value,h)
                m.append(mask)
            mask_l = np.array(m) & mask_l
        # re_sult = result[mask_l]
        if mask_l is None:
            print('M')
        return  mask_l[0,:]

class get_results:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def cluster(self):
        return

    def decoder(self):
        return

    def P_to_T(self,fea_m_size , stride=1):
        '''将坐标对应起来'''

        h, w = fea_m_size
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = torch.reshape(shift_x, [-1])
        shift_y = torch.reshape(shift_y, [-1])
        # 0x1y
        coords = torch.stack([shift_y, shift_x], -1)
        return coords.reshape(h,w,2).to(self.device)
    def resize_map(self,map):
        batch, c, h, w = map.shape  # detection batch=1
        map = map.reshape(c, h, w)
        map = map.permute(1, 2, 0).contiguous()

        return map
    def _nms(self, heat, kernel=5):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep
    def center_map(self,offset_map,cls_map):
        # print(len(offset_map.size()),offset_map.size(),offset_map.shape)
        if len(offset_map.size())==4:
            cls_map = self.resize_map(cls_map)
            offset_map = self.resize_map(offset_map)
        h,w = cls_map.shape[:2]
        mask_cls = cls_map > 0.25
        '''这地方有问题，多类怎么处理，最好做成 h,w,2的形式，0conf 1cls'''
        mask_cls = mask_cls.reshape(h,w)
        offset_map_1 = offset_map[mask_cls][:,[1,0]]
        candidate_map = offset_map_1+self.P_to_T((h,w))[mask_cls]
        zer = np.zeros((h, w))
        candidate_map = candidate_map.round()
        candidate_out, indices, counts = torch.unique(candidate_map, sorted=False, return_inverse=True, return_counts=True, dim=0)


        mask_candidate = counts > 0
        final_can = candidate_out[mask_candidate]
        final_can[:,0] = torch.clip(final_can[:,0],min=0,max=h-1)
        final_can[:, 1] = torch.clip(final_can[:, 1], min=0, max=w - 1)
        final_count = counts[mask_candidate]
        for i in range(0,final_can.size(0)):
            y = final_can[i][0]
            x = final_can[i][1]
            zer[int(y),int(x)] = final_count[i]
        # print(zer[zer>0])
        zer_t = torch.from_numpy(zer).resize(1,1,h,w)
        # print(zer_t)
        zer_t = self._nms(zer_t)
        # print(zer_t.shape)
        zer_t_n = np.array(zer_t.reshape(h,w))

        center_mask = zer_t_n > 10
        # print(self.P_to_T((h,w))[:,:,[1,0]][center_mask])

        c_m = cls_map.cpu()
        heat = np.array(c_m.reshape(h,w))
        # Heatmap_Show().MATPLOT_SHOW(heatmap=heat, Multiple=False)
        # Heatmap_Show().MATPLOT_SHOW(heatmap=zer, Multiple=False)
        #
        mask_zer = zer>0
        Heat_zer = heat.copy()

        # from matplotlib import cm
        # from matplotlib.ticker import LinearLocator, FormatStrFormatter
        #
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        #
        # # Make data.
        # X = np.arange(0, w, 1)
        # Y = np.arange(0, h ,1)
        # X, Y = np.meshgrid(X, Y)
        #
        #
        # surf = ax.plot_surface(Y, X, zer, cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        #
        # # Customize the z axis.
        # # ax.set_zlim(0, 255)  # z轴的取值范围
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #
        # # Add a color bar which maps values to colors.
        # # fig.colorbar(surf, shrink=0.5, aspect=5)
        #
        # plt.show()

        Heat_zer[zer>0] = 2
        # Heatmap_Show().MATPLOT_SHOW(heatmap=Heat_zer, Multiple=False)
        # heat[(zer_t_n<21)&(heat>0.25)] = 0.2#阈值设为20
        # print(heat[zer_t_n>20],zer_t_n[zer_t_n>1])
        # zer = zer + heat
        # Heatmap_Show().MATPLOT_SHOW(heatmap=heat,Multiple=False)


        return center_mask,zer_t_n
    #需要分析一下什么样的点才会刷到外面去
    def center_mask_results(self):
        '''验证一下是不是中心点附近区域的点预测到的值加准确？？？'''
        return

    def cycle_area_pred(self, pr_decs):
        '''decoder'''
        heat_map = self.resize_map(pr_decs['hm'])# h w c
        REG = self.resize_map(pr_decs['REG'])# h w 6
        theta_cls = self.resize_map(pr_decs['theta_cls'])# h w 18
        theta_reg = self.resize_map(pr_decs['theta_reg'])# h w 1
        h, w = heat_map.shape[:2]
        size = REG[:, :, 0:2]
        center_offsets = REG[:, :, 2:4]
        offsets = REG[:, :, 4:]

        center_mask,zer_t_n = self.center_map(center_offsets,heat_map)
        cls_pred = heat_map[center_mask]#[n,c]
        cls_score, cls_ind = torch.max(cls_pred, dim=-1)
        cls_score, cls_ind = cls_score.resize(cls_score.shape[0],1), cls_ind.resize(cls_score.shape[0],1)
        theta_cls_pred = theta_cls[center_mask]
        theta_cls_pred_max, theta_cls_pred_ind = torch.max(theta_cls_pred, dim=-1)
        theta_cls_pred_ind =theta_cls_pred_ind.resize(cls_score.shape[0], 1)
        theta_reg_pred = theta_reg[center_mask]
        cls_theta = theta_cls_pred_ind * 10 + theta_reg_pred


        center_coor = self.P_to_T((h,w))[:,:,[1,0]][center_mask]

        # print(center_coor)
        size_pred = size[center_mask]

        center_coor = center_coor + offsets[center_mask]


        detections = torch.cat([center_coor,size_pred,cls_theta,cls_score,cls_ind], dim=1)
        '''类内解耦'''
        SCR = screen(detections)
        mask_res = SCR.main(center_coor, zer_t_n)
        detections = detections[mask_res]
        detections = detections.unsqueeze(0)
        # print(detections.shape)


        # print(SCR.surrounding_SUM(,zer_t_n))
        return detections


# from pooling import rthombus
class Final_Treatment:

    '''后处理类，该类用于后处理步骤，其中包括 预测出的中心点n领域偏执，对角度进行左转右转等操作'''
    def __init__(self,conf_map):
        self.Decoder_reg = Decoder_REG_Targets()
        self.conf_map = conf_map
        self.map_size = (conf_map.shape[0],conf_map.shape[1])

    def rthombus(self,box):
        # (4x2)
        c12 = (box[0, :] + box[1, :]) / 2
        c23 = (box[1, :] + box[2, :]) / 2
        c34 = (box[2, :] + box[3, :]) / 2
        c14 = (box[0, :] + box[3, :]) / 2
        contour = np.concatenate([c12, c23, c34, c14], axis=0).reshape(4, 2)
        return contour
    def mask_conf(self,results):
        '''该函数用于解析预测对象的置信度'''
        im = np.zeros((self.map_size[0], self.map_size[1]), dtype="uint8")
        center_x, center_y, all_w, all_h, all_theta, all_cls, all_conf = results
        cbbx = self.Decoder_reg.decoder_reg(
            (center_x, center_y), (all_w, all_h), all_theta)
        contour = self.rthombus(cbbx)
        cv2.polylines(im, [contour.astype(np.int32)], 1, 255)
        cv2.fillPoly(im, [contour.astype(np.int32)], 255)
        mask = im == 255
        num_point = (im[mask].shape[0])
        num_sc = (np.sum(self.conf_map[mask]))
        if num_point == 0:
            score = 0
        else:
            score = num_sc / num_point
        return score

    def results_angle(self,result,ratio=20):
        '''ratio 左右偏转的角度'''
        center_x, center_y, all_w, all_h, all_theta, all_cls, all_conf = result
        results = []
        for i in range(0,ratio*2+1):
            if i>ratio-1:
                degree = ratio - i
            else:
                degree = i + 1
            re_a = [center_x, center_y, all_w, all_h, all_theta+degree, all_cls, all_conf]
            results.append(re_a)
        return results
    def results_center(self,result,parm=2):
        center_x, center_y, all_w, all_h, all_theta, all_cls, all_conf = result
        results = []
        parm_l1 = [[-1,-1],[1,1],[0,1],[0,-1],[1,-1],[-1,1],[1,0],[-1,0]]
        parm_l2_t = [[-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [-2, -1], [2, -1], [-2, 0],[2,0],[-2,1],[2,1],[-2,2],[2,2],[-1,2],[1,2],[0,2]]
        if parm == 1:
            parm_list = parm_l1
        else:
            parm_list = parm_l1+parm_l2_t
        for i in parm_list:
            i_x,i_y = i
            re_a = [center_x+i_x, center_y+i_y, all_w, all_h, all_theta, all_cls, all_conf]
            results.append(re_a)
        return results

    def main(self,result):
        result_score = self.mask_conf(result)
        if result_score > 0.6:
            return result
        else:
            results_a = self.results_angle(result)
            scores = []
            for re in results_a:
                scores.append(self.mask_conf(re))
            value = max(scores)  # 最大值
            idx = scores.index(value)
            # return results_a[idx]
            result_t = results_a[idx]
            results_t = self.results_center(result_t)
            scores = []
            for re in results_t:
                scores.append(self.mask_conf(re))
            value = max(scores)  # 最大值
            idx = scores.index(value)
            return results_t[idx]

