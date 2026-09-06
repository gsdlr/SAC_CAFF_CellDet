# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2023/03/03
"""

'''
dataset split
'''
# split = {'VGG': 100, 'ADI': 100, 'MBM': 34, 'MC': 50, 'Collected': 30, 'BCD': 560, 'PSU': 100, 'IMM': 100}


'''
batch size
'''
bs = {'VGG_100%': 4, 'VGG_10%': 4, 'MBM': 2, 'UniCD_50%': 4, 'BCD_50%': 2, 'BCD_10%': 2, 'PSU': 2, 'IMM': 4}

"""
scale
"""
scale = {'VGG_semi': 100.0, 'VGG_10%': 100.0,
         'UniCD_10%': 100.0, 'UniCD_20%': 100.0, 'UniCD_30%': 100.0, 'UniCD_50%': 100.0, 'UniCD_100%': 100.0,
         'BCD_10%': 100.0, 'BCD_20%': 100.0, 'BCD_30%': 100.0, 'BCD_50%': 100.0,'BCD_20%_train_320': 100.0,'BCD': 100.0,
         'PanNuke_10%': 100.0, 'PanNuke_20%': 100.0, 'PanNuke_30%': 100.0, 'PanNuke_50%': 100.0, 'PanNuke_100%': 100.0,
         }

'''
When the distance between the given predicted point pred_p and ground truth point gt_p is less than a distance threshold r, 
it means the pred_p and gt_p are successfully matched. The threshold r is related to real cell size. 
'''
radius = {
            'S': {'VGG_semi': 4, 'VGG_10%': 4, 'IMM': 8,
                  'UniCD_10%': 8, 'UniCD_20%': 8, 'UniCD_30%': 8, 'UniCD_50%': 8, 'UniCD_100%': 8,
                  'BCD_10%': 10, 'BCD_20%': 10, 'BCD_30%': 10, 'BCD_50%': 10,'BCD_20%_train_320': 10,'BCD': 10,
                  'PanNuke_10%': 12, 'PanNuke_20%': 12, 'PanNuke_30%': 12, 'PanNuke_50%': 12, 'PanNuke_100%': 12,
                  },


            'L': {'VGG': 8, 'ADI': 20, 'MC': 16, 'Collected': 16}
          }