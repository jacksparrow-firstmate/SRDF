import torch.utils.data as data
import torch,cv2
import numpy as np
class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None

        # self.image_distort = data_augment.PhotometricDistort()
        # self.IM_A = IM_AUG()

    def load_img_ids(self):
        """
        Definition: generate self.img_ids
        Usage: index the image properties (e.g. image name) for training, testing and evaluation
        Format: self.img_ids = [list]
        Return: self.img_ids
        """
        return None

    def load_image(self, index):
        """
        Definition: read images online
        Input: index, the index of the image in self.img_ids
        Return: image with H x W x 3 format
        """
        return None

    def load_annoFolder(self, img_id):
        """
        Return: the path of annotation
        Note: You may not need this function
        """
        return None

    def load_annotation(self, index):
        """
        Return: dictionary of {'pts': float np array of [bl, tl, tr, br],
                                'cat': int np array of class_index}
        Explaination:
                bl: bottom left point of the bounding box, format [x, y]
                tl: top left point of the bounding box, format [x, y]
                tr: top right point of the bounding box, format [x, y]
                br: bottom right point of the bounding box, format [x, y]
                class_index: the category index in self.category
                    example: self.category = ['ship]
                             class_index of ship = 0
        """
        return None

    def dec_evaluation(self, result_path):
        return None

    def __len__(self):
        return len(self.img_ids)
    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image
    def __getitem__(self, index):
        image = self.load_image(index)
        image_h, image_w, c = image.shape
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}