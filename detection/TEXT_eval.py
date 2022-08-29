from __future__ import print_function

import os
import torch
import zipfile
import shutil
import numpy as np

def sort_points_clockwise(PointList):

    points=np.array(PointList)
    outter_rect_l_t = np.append(np.min(points[::, 0]), np.min(points[::, 1]))

    l_t_point_index = np.argmin(
        np.sum(np.square(points - outter_rect_l_t), axis=1))

    l_t_point = points[l_t_point_index]
    other_three_points = np.append(points[0:l_t_point_index:],
                                   points[l_t_point_index + 1::],
                                   axis=0)

    BASE_VECTOR = np.asarray((1, 0))
    BASE_VECTOR_NORM = 1.0  # np.linalg.norm((1, 0))结果为1
    other_three_points = sorted(other_three_points,
                                key=lambda item: np.arccos(
                                    np.dot(BASE_VECTOR, item) /
                                    (BASE_VECTOR_NORM * np.linalg.norm(item))),
                                reverse=False)
    sorted_points = np.append(l_t_point.reshape(-1, 2),
                              np.asarray(other_three_points),
                              axis=0)
    return sorted_points



def icdar_evaluate(output=''):


    result = os.popen('cd {0} && python script.py -g=gt.zip -s=submit.zip '.format(output)).read()
    print(result)
    sep = result.split(':')
    precision = sep[1][:sep[1].find(',')].strip()
    recall = sep[2][:sep[2].find(',')].strip()
    f1 = sep[3][:sep[3].find(',')].strip()
    map = 0
    p = eval(precision)
    r = eval(recall)
    hmean = eval(f1)
    # display result
    print((p, r, 0, hmean))
    return p, r, map, hmean



if __name__ == '__main__':
    # generate_sumbzip()
    # generate_gtzip()
    icdar_evaluate(output=r'')
