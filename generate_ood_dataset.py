import os
import shutil
from collections import OrderedDict
import json

image_root_path = "/datasets/imagenet21k_resized"
maks_root_path = "/datasets/imagenet21k_resized/OOD_imagenet_seg_pseudomasks_clipseg"
train_iid_txt_path = "/datasets/imagenet21k_resized/lists/classification/test_hard_ood.txt" #"/mnt/gpu3_datasets/lists/segmentation/train_iid.txt" # "/mnt/gpu3_datasets/lists/segmentation/test_hard_ood.txt" # "/mnt/gpu3_datasets/lists/segmentation/test_easy_ood.txt"

file = open(train_iid_txt_path,'r')
label_to_name = dict()
data_statistic = dict()

semantic_label = {1: 'bag', 2: 'ball', 3: 'bed', 4: 'beer', 5: 'berry', 6: 'bird', 7: 'boat', 8: 'bottle', 9: 'bread', 10: 'cake', 11: 'camera', 12: 'candy', 13: 'car', 14: 'castle', 15: 'cat', 16: 'chair', 17: 'chicken', 18: 'cocktail', 19: 'coffee', 20: 'convenience_store', 21: 'cosmetics', 22: 'dog', 23: 'dress', 24: 'fish', 25: 'flower', 26: 'footwear', 27: 'frog', 28: 'furniture', 29: 'gun', 30: 'hat', 31: 'helmet', 32: 'horse', 33: 'house', 34: 'insect', 35: 'jacket', 36: 'knife', 37: 'lamp', 38: 'lizard', 39: 'medical_equipment', 40: 'musical_instrument', 41: 'person', 42: 'plane', 43: 'pot', 44: 'printer', 45: 'sandwich', 46: 'seafood', 47: 'skirt', 48: 'snake', 49: 'table', 50: 'telephone', 51: 'tree', 52: 'trousers', 53: 'truck', 54: 'turtle', 55: 'vegetable', 56: 'wine'}

# classification__label =  {1: 'bag', 2: 'ball', 3: 'bed', 4: 'beer', 5: 'berry', 6: 'bird', 7: 'boat', 8: 'bottle', 9: 'bread', 10: 'cake', 11: 'camera', 12: 'candy', 13: 'car', 14: 'castle', 15: 'cat', 16: 'chair', 17: 'chicken', 18: 'cocktail', 19: 'coffee', 20: 'convenience_store', 21: 'cosmetics', 22: 'dog', 23: 'dress', 24: 'fish', 25: 'flower', 26: 'footwear', 27: 'frog', 28: 'furniture', 29: 'gun', 30: 'hat', 31: 'helmet', 32: 'horse', 33: 'house', 34: 'insect', 35: 'jacket', 36: 'knife', 37: 'lamp', 38: 'lizard', 39: 'medical_equipment', 40: 'musical_instrument', 41: 'person', 42: 'plane', 43: 'pot', 44: 'printer', 45: 'sandwich', 46: 'seafood', 47: 'skirt', 48: 'snake', 49: 'table', 50: 'telephone', 51: 'tree', 52: 'trousers', 53: 'truck', 54: 'turtle', 55: 'vegetable', 56: 'wine'}

selected_label = OrderedDict({
# 1: 'bag',
# 6: 'bird',
7: 'boat',
13: 'car',
15: 'cat',
22: 'dog',
41: 'person',
42: 'plane',
# 53: 'truck',
# 54: 'turtle'
})


all_labels = set()
final_data = []

while True:
    content = file.readline().strip()
    if not content:
        break

    temp = content.split(" ")

    ''' segmentation '''
    if "segmentation" in train_iid_txt_path:
        image_path = temp[0]
        mask_path = temp[1]
        label = int(temp[2])
        label_name = temp[3]

    ''' classification '''
    if "classification" in train_iid_txt_path:
        image_path = temp[0]
        label = int(temp[1])
        label_name = temp[2]
        mask_path = None

    # img_full_path = os.path.join(image_root_path, image_path)

    all_labels.add(label)

    # if label not in selected_label.keys():
    #     continue

    if label not in label_to_name.keys():
        label_to_name[label] = label_name

    # if os.path.exists(img_full_path):
    if label not in data_statistic.keys():
        data_statistic[label] = 1
    else:
        data_statistic[label] += 1
    # else:
    #     print("no images")

    data = dict()
    data["image_path"] = image_path

    ''' semantic '''
    data["mask_path"] = mask_path

    ''' classification'''
    data["label"] = label
    data["label_name"] = label_to_name[label]

    final_data.append(data)

# with open('jsons/imagenet_sood_train_entire_segmentation.json', 'a') as f:
#         json.dump(final_data, f)

# with open('jsons/imagenet_sood_test_easy_ood_entire_classification.json', 'a') as f:
#     json.dump(final_data, f)

with open('jsons/imagenet_sood_test_hard_ood_entire_classification.json', 'a') as f:
    json.dump(final_data, f)


print("all label: ", all_labels)
print("label_to_name:", label_to_name)
print("data_statistic:", data_statistic, sum(n for n in data_statistic.values()))

# entire images classification
# all label:  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
# label_to_name: {0: 'bag', 1: 'ball', 2: 'bed', 3: 'beer', 4: 'berry', 5: 'bird', 6: 'boat', 7: 'bottle', 8: 'bread', 9: 'cake', 10: 'camera', 11: 'candy', 12: 'car', 13: 'castle', 14: 'cat', 15: 'chair', 16: 'chicken', 17: 'cocktail', 18: 'coffee', 19: 'convenience_store', 20: 'cosmetics', 21: 'dog', 22: 'dress', 23: 'fish', 24: 'flower', 25: 'footwear', 26: 'frog', 27: 'furniture', 28: 'gun', 29: 'hat', 30: 'helmet', 31: 'horse', 32: 'house', 33: 'insect', 34: 'jacket', 35: 'knife', 36: 'lamp', 37: 'lizard', 38: 'medical_equipment', 39: 'musical_instrument', 40: 'person', 41: 'plane', 42: 'pot', 43: 'printer', 44: 'sandwich', 45: 'seafood', 46: 'skirt', 47: 'snake', 48: 'table', 49: 'telephone', 50: 'tree', 51: 'trousers', 52: 'truck', 53: 'turtle', 54: 'vegetable', 55: 'wine'}
# data_statistic: {0: 21482, 1: 15211, 2: 11108, 3: 11940, 4: 11263, 5: 32199, 6: 24223, 7: 10242, 8: 19040, 9: 10718, 10: 8366, 11: 11221, 12: 22627, 13: 8334, 14: 17229, 15: 12540, 16: 6955, 17: 8520, 18: 6767, 19: 7523, 20: 15175, 21: 51301, 22: 20917, 23: 21157, 24: 120837, 25: 21028, 26: 16981, 27: 25580, 28: 11737, 29: 14406, 30: 7112, 31: 30519, 32: 17019, 33: 26888, 34: 7282, 35: 14584, 36: 14173, 37: 7467, 38: 14821, 39: 25877, 40: 27992, 41: 12197, 42: 10107, 43: 7673, 44: 9200, 45: 19801, 46: 7225, 47: 20448, 48: 20774, 49: 8132, 50: 41028, 51: 12552, 52: 9762, 53: 16522, 54: 21112, 55: 31335} 1038229

# easy ood classification
# all label:  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
# label_to_name: {0: 'bag', 1: 'ball', 2: 'bed', 3: 'beer', 4: 'berry', 5: 'bird', 6: 'boat', 7: 'bottle', 8: 'bread', 9: 'cake', 10: 'camera', 11: 'candy', 12: 'car', 13: 'castle', 14: 'cat', 15: 'chair', 16: 'chicken', 17: 'cocktail', 18: 'coffee', 19: 'convenience_store', 20: 'cosmetics', 21: 'dog', 22: 'dress', 23: 'fish', 24: 'flower', 25: 'footwear', 26: 'frog', 27: 'furniture', 28: 'gun', 29: 'hat', 30: 'helmet', 31: 'horse', 32: 'house', 33: 'insect', 34: 'jacket', 35: 'knife', 36: 'lamp', 37: 'lizard', 38: 'medical_equipment', 39: 'musical_instrument', 40: 'person', 41: 'plane', 42: 'pot', 43: 'printer', 44: 'sandwich', 45: 'seafood', 46: 'skirt', 47: 'snake', 48: 'table', 49: 'telephone', 50: 'tree', 51: 'trousers', 52: 'truck', 53: 'turtle', 54: 'vegetable', 55: 'wine'}
# data_statistic: {0: 7085, 1: 3427, 2: 2477, 3: 3671, 4: 4066, 5: 10205, 6: 7370, 7: 4207, 8: 6839, 9: 3211, 10: 3411, 11: 3846, 12: 7888, 13: 2237, 14: 5024, 15: 4977, 16: 2136, 17: 2760, 18: 2172, 19: 2428, 20: 4699, 21: 16335, 22: 6339, 23: 6454, 24: 40582, 25: 6778, 26: 4767, 27: 9224, 28: 4364, 29: 5244, 30: 2748, 31: 10733, 32: 5653, 33: 8978, 34: 3171, 35: 5111, 36: 5753, 37: 1191, 38: 3775, 39: 7892, 40: 9391, 41: 4375, 42: 2609, 43: 2101, 44: 3324, 45: 6028, 46: 2736, 47: 6279, 48: 6712, 49: 2483, 50: 13446, 51: 3976, 52: 3840, 53: 4518, 54: 5900, 55: 11247} 338193

# hard ood classification
# all label:  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
# label_to_name: {0: 'bag', 1: 'ball', 2: 'bed', 3: 'beer', 4: 'berry', 5: 'bird', 16: 'chicken', 6: 'boat', 7: 'bottle', 8: 'bread', 9: 'cake', 10: 'camera', 11: 'candy', 12: 'car', 13: 'castle', 14: 'cat', 15: 'chair', 17: 'cocktail', 18: 'coffee', 19: 'convenience_store', 20: 'cosmetics', 21: 'dog', 22: 'dress', 23: 'fish', 24: 'flower', 25: 'footwear', 26: 'frog', 27: 'furniture', 48: 'table', 28: 'gun', 29: 'hat', 30: 'helmet', 31: 'horse', 32: 'house', 33: 'insect', 34: 'jacket', 35: 'knife', 36: 'lamp', 37: 'lizard', 38: 'medical_equipment', 39: 'musical_instrument', 40: 'person', 41: 'plane', 42: 'pot', 43: 'printer', 44: 'sandwich', 45: 'seafood', 46: 'skirt', 47: 'snake', 49: 'telephone', 50: 'tree', 51: 'trousers', 52: 'truck', 53: 'turtle', 54: 'vegetable', 55: 'wine'}
# data_statistic: {0: 6950, 1: 4521, 2: 3006, 3: 2349, 4: 3349, 5: 10159, 16: 2246, 6: 7703, 7: 2294, 8: 5333, 9: 2612, 10: 1589, 11: 3217, 12: 6501, 13: 1606, 14: 4667, 15: 3275, 17: 2624, 18: 1666, 19: 1651, 20: 3574, 21: 15269, 22: 5947, 23: 6839, 24: 39888, 25: 5916, 26: 4782, 27: 7454, 48: 5949, 28: 3423, 29: 3974, 30: 1316, 31: 9137, 32: 4518, 33: 8098, 34: 1642, 35: 3954, 36: 3094, 37: 2049, 38: 4356, 39: 7181, 40: 8789, 41: 3575, 42: 2217, 43: 1983, 44: 2544, 45: 5510, 46: 1077, 47: 5874, 49: 1835, 50: 12422, 51: 3366, 52: 2406, 53: 5142, 54: 6550, 55: 9356} 298324

# entire images semantic
# all label:  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56}
# label_to_name: {1: 'bag', 2: 'ball', 3: 'bed', 4: 'beer', 5: 'berry', 6: 'bird', 7: 'boat', 8: 'bottle', 9: 'bread', 10: 'cake', 11: 'camera', 12: 'candy', 13: 'car', 14: 'castle', 15: 'cat', 16: 'chair', 17: 'chicken', 18: 'cocktail', 19: 'coffee', 20: 'convenience_store', 21: 'cosmetics', 22: 'dog', 23: 'dress', 24: 'fish', 25: 'flower', 26: 'footwear', 27: 'frog', 28: 'furniture', 29: 'gun', 30: 'hat', 31: 'helmet', 32: 'horse', 33: 'house', 34: 'insect', 35: 'jacket', 36: 'knife', 37: 'lamp', 38: 'lizard', 39: 'medical_equipment', 40: 'musical_instrument', 41: 'person', 42: 'plane', 43: 'pot', 44: 'printer', 45: 'sandwich', 46: 'seafood', 47: 'skirt', 48: 'snake', 49: 'table', 50: 'telephone', 51: 'tree', 52: 'trousers', 53: 'truck', 54: 'turtle', 55: 'vegetable', 56: 'wine'}
# data_statistic: {1: 21482, 2: 15211, 3: 11108, 4: 11940, 5: 11263, 6: 32199, 7: 24223, 8: 10242, 9: 19040, 10: 10718, 11: 8366, 12: 11221, 13: 22627, 14: 8334, 15: 17229, 16: 12540, 17: 6955, 18: 8520, 19: 6767, 20: 7523, 21: 15175, 22: 51301, 23: 20917, 24: 21157, 25: 120837, 26: 21028, 27: 16981, 28: 25580, 29: 11737, 30: 14406, 31: 7112, 32: 30519, 33: 17019, 34: 26888, 35: 7282, 36: 14584, 37: 14173, 38: 7467, 39: 14821, 40: 25877, 41: 27992, 42: 12197, 43: 10107, 44: 7673, 45: 9200, 46: 19801, 47: 7225, 48: 20448, 49: 20774, 50: 8132, 51: 41028, 52: 12552, 53: 9762, 54: 16522, 55: 21112, 56: 30714} 1037608

# easy ood semantic
# all label:  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56}
# label_to_name: {1: 'bag', 2: 'ball', 3: 'bed', 4: 'beer', 5: 'berry', 6: 'bird', 7: 'boat', 8: 'bottle', 9: 'bread', 10: 'cake', 11: 'camera', 12: 'candy', 13: 'car', 14: 'castle', 15: 'cat', 16: 'chair', 17: 'chicken', 18: 'cocktail', 19: 'coffee', 20: 'convenience_store', 21: 'cosmetics', 22: 'dog', 23: 'dress', 24: 'fish', 25: 'flower', 26: 'footwear', 27: 'frog', 28: 'furniture', 29: 'gun', 30: 'hat', 31: 'helmet', 32: 'horse', 33: 'house', 34: 'insect', 35: 'jacket', 36: 'knife', 37: 'lamp', 38: 'lizard', 39: 'medical_equipment', 40: 'musical_instrument', 41: 'person', 42: 'plane', 43: 'pot', 44: 'printer', 45: 'sandwich', 46: 'seafood', 47: 'skirt', 48: 'snake', 49: 'table', 50: 'telephone', 51: 'tree', 52: 'trousers', 53: 'truck', 54: 'turtle', 55: 'vegetable', 56: 'wine'}
# data_statistic: {1: 103, 2: 58, 3: 25, 4: 110, 5: 86, 6: 114, 7: 96, 8: 98, 9: 92, 10: 68, 11: 101, 12: 51, 13: 82, 14: 75, 15: 97, 16: 91, 17: 104, 18: 73, 19: 75, 20: 8, 21: 84, 22: 107, 23: 75, 24: 95, 25: 67, 26: 95, 27: 111, 28: 52, 29: 90, 30: 98, 31: 93, 32: 100, 33: 68, 34: 94, 35: 69, 36: 63, 37: 59, 38: 85, 39: 31, 40: 59, 41: 69, 42: 93, 43: 49, 44: 93, 45: 60, 46: 79, 47: 69, 48: 85, 49: 47, 50: 92, 51: 23, 52: 96, 53: 85, 54: 97, 55: 46, 56: 100} 4385

# hard ood semantic
# all label:  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56}
# label_to_name: {1: 'bag', 2: 'ball', 3: 'bed', 4: 'beer', 5: 'berry', 6: 'bird', 17: 'chicken', 7: 'boat', 8: 'bottle', 9: 'bread', 10: 'cake', 11: 'camera', 12: 'candy', 13: 'car', 14: 'castle', 15: 'cat', 16: 'chair', 18: 'cocktail', 19: 'coffee', 20: 'convenience_store', 21: 'cosmetics', 22: 'dog', 23: 'dress', 24: 'fish', 25: 'flower', 26: 'footwear', 27: 'frog', 28: 'furniture', 49: 'table', 29: 'gun', 30: 'hat', 31: 'helmet', 32: 'horse', 33: 'house', 34: 'insect', 35: 'jacket', 36: 'knife', 37: 'lamp', 38: 'lizard', 39: 'medical_equipment', 40: 'musical_instrument', 41: 'person', 42: 'plane', 43: 'pot', 44: 'printer', 45: 'sandwich', 46: 'seafood', 47: 'skirt', 48: 'snake', 50: 'telephone', 51: 'tree', 52: 'trousers', 53: 'truck', 54: 'turtle', 55: 'vegetable', 56: 'wine'}
# data_statistic: {1: 87, 2: 79, 3: 41, 4: 97, 5: 68, 6: 92, 17: 92, 7: 73, 8: 90, 9: 73, 10: 44, 11: 92, 12: 73, 13: 62, 14: 62, 15: 107, 16: 46, 18: 73, 19: 40, 20: 15, 21: 86, 22: 96, 23: 51, 24: 76, 25: 53, 26: 76, 27: 94, 28: 34, 49: 54, 29: 61, 30: 87, 31: 98, 32: 92, 33: 57, 34: 83, 35: 63, 36: 81, 37: 51, 38: 89, 39: 41, 40: 41, 41: 74, 42: 78, 43: 85, 44: 86, 45: 73, 46: 60, 47: 66, 48: 88, 50: 66, 51: 18, 52: 72, 53: 80, 54: 90, 55: 52, 56: 92} 3950

