import pickle
import numpy as np
import sys

# file = '/Users/justinbutler/Desktop/GPU_Server_Stuff/detectron_mask/anchor_log_test_ones/anchor_data.pickle'
# file = '/home/justin/Models/detectron2/mask_paper_4/anchor_data.pickle'
# file = '/home/justin/Models/Yolo/mask_paper_4/anchor_data.pickle'
file = sys.argv[1]
print(f'Reading file: {file}')

data = []
with open(file, 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass

total_step_count = 0
count = 0
small_avg_iou = []
med_avg_iou = []
large_avg_iou = []
num_pos_anchors = []
small_pos_match = []
med_pos_match = []
large_pos_match = []
num_gt_objects = []
num_small_gt = []
num_med_gt = []
num_large_gt = []
small_iou_1 = []
small_iou_2 = []
small_iou_3 = []
small_iou_4 = []
small_iou_5 = []
small_iou_6 = []
small_iou_7 = []
small_iou_8 = []
small_iou_9 = []
small_iou_10 = []

medium_iou_1 = []
medium_iou_2 = []
medium_iou_3 = []
medium_iou_4 = []
medium_iou_5 = []
medium_iou_6 = []
medium_iou_7 = []

large_iou_1 = []
large_iou_2 = []
large_iou_3 = []
large_iou_4 = []
large_iou_5 = []
large_iou_6 = []
large_iou_7 = []
small_zero_values = []
medium_zero_values = []
large_zero_values = []

# print(data)

for item in data:
    print(item)
    print(f'count: {count}')
    if count == 0:
        small_pos_match.append(item)
        count += 1
    elif count == 1:
        med_pos_match.append(item)
        count += 1
    elif count == 2:
        large_pos_match.append(item)
        count += 1
    elif count == 3:
        num_small_gt.append(item)
        count += 1
    elif count == 4:
        num_med_gt.append(item)
        count += 1
    elif count == 5:
        num_large_gt.append(item)
        count += 1

    if count == 6:
        count = 0

# Totals
total_pos_anchors = np.sum(small_pos_match) + np.sum(med_pos_match) + np.sum(large_pos_match)
# Num Pos Anchors data
small_match = np.sum(small_pos_match) / total_pos_anchors * 100
med_match = np.sum(med_pos_match) / total_pos_anchors * 100
large_match = np.sum(large_pos_match) / total_pos_anchors * 100

print('Number Positive Anchors Per Size in Percent')
print(small_match)
print(med_match)
print(large_match)

print(f'Total Pos Anchors: {total_pos_anchors}')

print(f'Total Small Pos. Anchors: {np.sum(small_pos_match)}')
print(f'Total small anchors: {np.sum(num_small_gt)}')

print(f'Total medium Pos. Anchors: {np.sum(med_pos_match)}')
print(f'Total medium anchors: {np.sum(num_med_gt)}')

print(f'Total large Pos. Anchors: {np.sum(large_pos_match)}')
print(f'Total large anchors: {np.sum(num_large_gt)}')
