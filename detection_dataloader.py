import numpy as np
np.set_printoptions(threshold=np.NaN)
import cv2

import copy
import os
import glob

import xml.etree.ElementTree as ET
import pickle

import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

##########################################################################################
##########################################################################################
#VOC data parse
def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(os.listdir(ann_dir)):
            img = {'object':[]}

            try:
                tree = ET.parse(ann_dir + ann)
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)
                continue
            
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
                            
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        #print(cache)
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels
def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    use_valid,
    labels):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")
        if use_valid:
            train_valid_split = int(0.8*len(train_ints))
            np.random.shuffle(train_ints)

            valid_ints = train_ints[train_valid_split:]
            train_ints = train_ints[:train_valid_split]
        else:
            valid_ints = []

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t\t'  + str(train_labels))
        print('Given labels: \t\t' + str(labels))
        print('Overlap labels: \t' + str(list(overlap_labels)))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    return train_ints, valid_ints, sorted(labels)

class BatchGenerator():
    def __init__(self,
        instances,
        labels,
        BatchSize,
        ssd_min_box_area=0.02,
        ssd_max_box_area=0.9,
        ssd_default_box_size=[6,6,6,6,6,6],
        ssd_default_box_ratio=[
                                [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
                                [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
                                [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
                                [1.0, 1.25, 2.0, 3.0, 1.0 / 2.0, 1.0 / 3.0],
                                [1.0, 1.25, 2.0, 3.0,1.0 / 2.0, 1.0 / 3.0],
                                [1.0, 1.25, 2.0, 3.0,1.0 / 2.0, 1.0 / 3.0]
                              ],
        ssd_downsample=[4,8,16,16,32,32],
        ssd_max_downsample=32,
        ssd_min_net_size=32*7,
        ssd_max_net_size=32*11,
        ssd_iou_thresh=0.6,
        shuffle = True
    ):
        self.instances = instances
        self.labels = labels
        self.shuffle = shuffle
        if shuffle: np.random.shuffle(self.instances)
        self.BatchSize = BatchSize
        
        self.ssd_default_box_area = np.linspace(ssd_min_box_area, ssd_max_box_area, num = np.amax(ssd_default_box_size))
        self.ssd_default_box_ratio = ssd_default_box_ratio
        self.ssd_downsample = ssd_downsample
        self.ssd_max_downsample = ssd_max_downsample
        self.ssd_min_net_size = ssd_min_net_size
        self.ssd_max_net_size = ssd_max_net_size
        self.ssd_iou_thresh = ssd_iou_thresh
    
        self.batch_count = 0
    def next(self):
        while True:
            x_batch, ground_truth, positive, negative = self.__getitem__(self.batch_count)
            self.batch_count += 1
            if self.batch_count*self.BatchSize>self.__len__():
                self.batch_count = 0
                if self.shuffle: np.random.shuffle(self.instances)
                print('------------------------------next epoch------------------------------')
            yield x_batch, ground_truth, positive, negative
    
    def __getitem__(self, idx):
        net_h, net_w = self._get_net_size(idx)
        
        l_round = idx*self.BatchSize
        r_round = (idx+1)*self.BatchSize
        if r_round>self.__len__():
            r_round = self.__len__()
            l_round = r_round-self.BatchSize
        
        x_batch = np.zeros((r_round-l_round,net_h,net_w,3), dtype=np.float32)
        actual_data = []
        
        all_default_boxs,num_anchors = self.generate_all_default_boxs(self.ssd_downsample,net_h)
        
        for i,train_instance in enumerate(self.instances[l_round:r_round]):
            img, _actual_data = self.__aug_image(train_instance,net_h,net_w)
            x_batch[i,:,:,:] = img
            actual_data.append(_actual_data)
        
        gt_class,gt_location,positive,negative = self.generate_groundtruth_data(actual_data,num_anchors,all_default_boxs)
        ground_truth = np.concatenate([gt_class[:,:,np.newaxis],gt_location],axis=-1)
        
        return x_batch, ground_truth, positive, negative
    def __aug_image(self, instance, net_h, net_w):
        def _constrain(min_v, max_v, value):
            if value < min_v: return min_v
            if value > max_v: return max_v
            return value
        
        image_name = instance['filename']
        image = cv2.imread(image_name)[:,:,::-1] # RGB image
        
        if image is None: print('Cannot find ', image_name)
            
        image_h, image_w, _ = image.shape
        
        im_sized = cv2.resize(image, (net_w, net_h))
        
        # randomly flip
        flip = np.random.randint(2)
        im_sized = self.random_flip(im_sized, flip)#随机翻转
        
        boxes = copy.deepcopy(instance['object'])

        # randomize boxes' order
        np.random.shuffle(boxes)

        # correct sizes and positions
        sx, sy = float(net_w)/image_w, float(net_h)/image_h
        zero_boxes = []
        _actual_data = []

        for i in range(len(boxes)):
            x_min = int(_constrain(0, net_w, boxes[i]['xmin']*sx))
            x_max = int(_constrain(0, net_w, boxes[i]['xmax']*sx))
            y_min = int(_constrain(0, net_h, boxes[i]['ymin']*sy))
            y_max = int(_constrain(0, net_h, boxes[i]['ymax']*sy))
            label = self.labels.index(boxes[i]['name'])
            #print(self.labels,label,boxes[i]['name'])
            _actual_data.append([((x_min + x_max)/2/net_w),((y_min + y_max)/2/net_h),((x_max - x_min)/net_w),((y_max - y_min)/net_h), label])

            if x_max <= x_min or y_max <= y_min:
                zero_boxes += [i]
                continue

            if flip == 1:
                _actual_data[i][0] = 1 - _actual_data[i][0]

        _actual_data = [_actual_data[i] for i in range(len(_actual_data)) if i not in zero_boxes]
        
        return im_sized, _actual_data
    def random_flip(self,image, flip):
        if flip == 1: return cv2.flip(image, 1)
        return image
    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = self.ssd_max_downsample*np.random.randint(self.ssd_min_net_size/self.ssd_max_downsample,self.ssd_max_net_size/self.ssd_max_downsample+1)
            print("resizing: ", net_size, net_size)
            self.net_size = net_size
        return self.net_size, self.net_size
    def __len__(self):
        return int(len(self.instances))
    
    def generate_groundtruth_data(self,input_actual_data,num_anchors,all_default_boxs):
        batch = len(input_actual_data)
        gt_class = np.zeros((batch, num_anchors)) 
        gt_location = np.zeros((batch, num_anchors, 4))
        gt_positives_iou = np.zeros((batch, num_anchors))
        gt_positives = np.zeros((batch, num_anchors))
        gt_negatives = np.zeros((batch, num_anchors))
        background_iou = max(0, (self.ssd_iou_thresh-0.2))
        # 初始化正例训练数据
        for img_index in range(batch):
            for pre_actual in input_actual_data[img_index]:
                gt_class_val = pre_actual[-1]
                gt_box_val = pre_actual[:-1]
                for boxe_index in range(num_anchors):
                    iou = self.iou(gt_box_val, all_default_boxs[boxe_index])
                    if iou >= self.ssd_iou_thresh:
                        gt_class[img_index][boxe_index] = gt_class_val
                        gt_location[img_index][boxe_index] = gt_box_val
                        gt_positives_iou[img_index][boxe_index] = iou
                        gt_positives[img_index][boxe_index] = 1
                        gt_negatives[img_index][boxe_index] = 0
            # 如果没有正例，则随机创建一个正例，预防nan
            if np.sum(gt_positives[img_index])==0 :
                print('[没有匹配iou]:'+str(input_actual_data[img_index]))
                random_pos_index = np.random.randint(low=0, high=num_anchors, size=1)[0]
                gt_class[img_index][random_pos_index] = 0
                gt_location[img_index][random_pos_index] = [0,0,0,0]
                gt_positives_iou[img_index][random_pos_index] = self.ssd_iou_thresh
                gt_positives[img_index][random_pos_index] = 1
                gt_negatives[img_index][random_pos_index] = 0
            # 正负例比值 1:3
            gt_neg_end_count = int(np.sum(gt_positives[img_index]) * 3)
            if (gt_neg_end_count+np.sum(gt_positives[img_index])) > num_anchors :
                gt_neg_end_count = num_anchors - np.sum(gt_positives[img_index])
            # 随机选择负例
            gt_neg_index = np.random.randint(low=0, high=num_anchors, size=gt_neg_end_count)
            for r_index in gt_neg_index:
                if gt_positives_iou[img_index][r_index] < background_iou : 
                    gt_class[img_index][r_index] = 0
                    gt_positives[img_index][r_index] = 0
                    gt_negatives[img_index][r_index] = 1
        return gt_class, gt_location, gt_positives, gt_negatives
    def iou(self, rect1, rect2):
        x_overlap = max(0, (min(rect1[0]+(rect1[2]/2), rect2[0]+(rect2[2]/2)) - max(rect1[0]-(rect1[2]/2), rect2[0]-(rect2[2]/2))))
        y_overlap = max(0, (min(rect1[1]+(rect1[3]/2), rect2[1]+(rect2[3]/2)) - max(rect1[1]-(rect1[3]/2), rect2[1]-(rect2[3]/2))))
        intersection = x_overlap * y_overlap
        # 删除超出图像大小的部分
        rect1_width_sub = 0
        rect1_height_sub = 0
        rect2_width_sub = 0
        rect2_height_sub = 0
        if (rect1[0]-rect1[2]/2) < 0 : rect1_width_sub += 0-(rect1[0]-rect1[2]/2)
        if (rect1[0]+rect1[2]/2) > 1 : rect1_width_sub += (rect1[0]+rect1[2]/2)-1
        if (rect1[1]-rect1[3]/2) < 0 : rect1_height_sub += 0-(rect1[1]-rect1[3]/2)
        if (rect1[1]+rect1[3]/2) > 1 : rect1_height_sub += (rect1[1]+rect1[3]/2)-1
        if (rect2[0]-rect2[2]/2) < 0 : rect2_width_sub += 0-(rect2[0]-rect2[2]/2)
        if (rect2[0]+rect2[2]/2) > 1 : rect2_width_sub += (rect2[0]+rect2[2]/2)-1
        if (rect2[1]-rect2[3]/2) < 0 : rect2_height_sub += 0-(rect2[1]-rect2[3]/2)
        if (rect2[1]+rect2[3]/2) > 1 : rect2_height_sub += (rect2[1]+rect2[3]/2)-1
        area_box_a = (rect1[2]-rect1_width_sub) * (rect1[3]-rect1_height_sub)
        area_box_b = (rect2[2]-rect2_width_sub) * (rect2[3]-rect2_height_sub)
        union = area_box_a + area_box_b - intersection
        if intersection > 0 and union > 0 : 
            return intersection / union 
        else : 
            return 0
    def generate_all_default_boxs(self,ssd_downsample,input_size):
        all_default_boxs = []
        num_anchors = 0
        for index, scale in enumerate(ssd_downsample):
            width = int(input_size/scale)
            height = int(input_size/scale)
            cell_scale = self.ssd_default_box_area[index]
            for x in range(width):
                for y in range(height):
                    for ratio in self.ssd_default_box_ratio[index]:
                        center_x = (x / float(width)) + (0.5/ float(width))
                        center_y = (y / float(height)) + (0.5 / float(height))
                        box_width = np.sqrt(cell_scale * ratio)
                        box_height = np.sqrt(cell_scale / ratio)
                        all_default_boxs.append([center_x, center_y, box_width, box_height])
                        num_anchors += 1
        all_default_boxs = np.array(all_default_boxs)
        all_default_boxs = self.check_numerics(all_default_boxs,'all_default_boxs') 
        return all_default_boxs,num_anchors
    def check_numerics(self, input_dataset, message):
        if str(input_dataset).find('Tensor') == 0 :
            input_dataset = tf.check_numerics(input_dataset, message)
        else :
            dataset = np.array(input_dataset)
            nan_count = np.count_nonzero(dataset != dataset) 
            inf_count = len(dataset[dataset == float("inf")])
            n_inf_count = len(dataset[dataset == float("-inf")])
            if nan_count>0 or inf_count>0 or n_inf_count>0:
                data_error = '【'+ message +'】出现数据错误！【nan：'+str(nan_count)+'|inf：'+str(inf_count)+'|-inf：'+str(n_inf_count)+'】'
                raise Exception(data_error) 
        return  input_dataset
##########################################################################################
##########################################################################################

if __name__=='__main__':
    train_ints, valid_ints, labels = create_training_instances(
                                                                'F:\\Learning\\tensorflow\\detect\\Dataset\\Fish\\Annotations\\',
                                                                'F:\\Learning\\tensorflow\\detect\\Dataset\\Fish\\JPEGImages\\',
                                                                'train',
                                                                '','','',False,
                                                                ['heidiao','niyu','lvqimamiantun','hualu','heijun','dalongliuxian','tiaoshiban']
                                                               )
    labels.insert(0,'BG')
    DataLoader = BatchGenerator(train_ints,labels,3)
    count = 0
    for x_batch, ground_truth, positive, negative in DataLoader.next():
        print(x_batch.shape,ground_truth.shape,positive.shape,negative.shape)
        count += 1
        if count>30:
            break