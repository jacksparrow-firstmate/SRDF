from datasets.base import BaseDataset
import os
import cv2
import numpy as np
from DOTA_devkit.ResultMerge_multi_process import mergebypoly
from datasets.eval_code import voc_eval

class DOTA(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(DOTA, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['plane',
                         'baseball-diamond',
                         'bridge',
                         'ground-track-field',
                         'small-vehicle',
                         'large-vehicle',
                         'ship',
                         'tennis-court',
                         'basketball-court',
                         'storage-tank',
                         'soccer-ball-field',
                         'roundabout',
                         'harbor',
                         'swimming-pool',
                         'helicopter'
                         ]
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        if self.phase == 'train':
            image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')
        else:
            image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.png')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                if len(obj)>8:
                    x1 = min(max(float(obj[0]), 0), w - 1)
                    y1 = min(max(float(obj[1]), 0), h - 1)
                    x2 = min(max(float(obj[2]), 0), w - 1)
                    y2 = min(max(float(obj[3]), 0), h - 1)
                    x3 = min(max(float(obj[4]), 0), w - 1)
                    y3 = min(max(float(obj[5]), 0), h - 1)
                    x4 = min(max(float(obj[6]), 0), w - 1)
                    y4 = min(max(float(obj[7]), 0), h - 1)

                    valid_pts.append([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
                    valid_cat.append(self.cat_ids[obj[8]])
                    valid_dif.append(int(obj[9]))
        f.close()
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)

        return annotation


    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)



class HRSC(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(HRSC, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['ship']
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'ALLIM')
        self.label_path = os.path.join(data_dir, 'ALLLAB')


    def load_img_ids(self):
        image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists


    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id + '.png')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id + '.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        valid_pts = []
        valid_cat = []
        valid_dif = []

        filename = self.load_annoFolder(self.img_ids[index])
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('\n')
            for obj in objects:
                if len(obj) != 0:
                    difficult = obj.split()[9]
                    box = obj.split(' ')[0:8]
                    label = self.cat_ids['ship']
                    box = [eval(x) for x in box]
                    valid_pts.append(box)
                    valid_cat.append(label)
                    valid_dif.append(difficult)

        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)

        return annotation

    def dec_evaluation(self, result_path):
        detpath = os.path.join(result_path, 'Task1_{}.txt')
        annopath = os.path.join(self.label_path, '{}.txt')  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
        imagesetfile = os.path.join(self.data_dir, 'test.txt')
        classaps = []
        map = 0
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            # print('{}:{} '.format(classname, ap*100))
            classaps.append(ap)
            # umcomment to show p-r curve of each category
            # plt.figure(figsize=(8,4))
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.plot(rec, prec)
        # plt.show()
        map = map / len(self.category)
        print('map:', map*100)
        # classaps = 100 * np.array(classaps)
        # print('classaps: ', classaps)
        return map
#
class TEXT(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(TEXT, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['text']
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'ALLIM')
        self.label_path = os.path.join(data_dir, 'ALLLAB')


    def load_img_ids(self):
        image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists


    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.jpg')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)

        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id + '.txt')
        # return os.path.join(self.label_path, 'gt_'+img_id + '.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape

        valid_pts = []
        valid_cat = []
        valid_dif = []
        filename = self.load_annoFolder(self.img_ids[index])
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('\n')
            for obj in objects:
                if len(obj) != 0:

                    box = obj.split(',')[0:8]
                    diff = obj.split(',')[-1]
                    if diff == '###':
                        difficult = 1
                    else:
                        difficult = 0
                    label = self.cat_ids['text']
                    box = [eval(x) for x in box]
                    valid_pts.append(box)
                    valid_cat.append(label)
                    valid_dif.append(difficult)

        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)

        return annotation

    def dec_evaluation(self, result_path):
        detpath = os.path.join(result_path, 'Task1_{}.txt')
        annopath = os.path.join(self.label_path, '{}.txt')  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
        imagesetfile = os.path.join(self.data_dir, 'test.txt')
        classaps = []
        map = 0
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            classaps.append(ap)
        map = map / len(self.category)
        print('map:', map*100)
        return map