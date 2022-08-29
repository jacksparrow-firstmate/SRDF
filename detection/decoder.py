import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from tools import get_results


class DecDecoder(object):
    def __init__(self,  conf_thresh, num_classes):

        self.conf_thresh = conf_thresh
        self.num_classes = num_classes


    def ctdet_decode(self, pr_decs):
        heat_map = pr_decs['hm']
        GR =get_results()
        detections1 = GR.cycle_area_pred(pr_decs)
        return detections1.data.cpu().numpy(),heat_map.data.cpu().numpy()