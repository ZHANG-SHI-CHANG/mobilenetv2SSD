import tensorflow as tf
from tensorflow.python.ops import array_ops

import numpy as np

Detection_or_Classifier = 'detection'#'detection','classifier'

class Mobilenetv2():
    
    def __init__(self,num_classes,learning_rate=0.00001):
        self.num_classes = num_classes
        
        self.ssd_default_box_size=[6,6,6,6,6,6]
        self.learning_rate = learning_rate
        
        self.loss = Loss()
        
        self.__build()
    
    def __build(self):
        self.norm = 'batch_norm'#group_norm,batch_norm
        self.activate = 'prelu'#selu,leaky,swish,relu,relu6,prelu
        self.BlockInfo = {#scale /8
                          '1':[1,16,1,1,True],
                          '2':[6,24,1,2,True],
                          '3':[6,24,1,1,True],#ssd
                          '4':[6,32,1,2,True],
                          '5':[6,32,2,1,True],#ssd
                          '6':[6,64,1,2,True],
                          '7':[6,64,3,1,True],#ssd
                          '8':[6,96,3,1,True],#ssd
                          '9':[6,160,1,2,True],
                          '10':[6,160,2,1,True],#ssd
                          '11':[6,320,1,1,True],#ssd
                          '12':[1,1280,1,1,False]}
    
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()
        
        with tf.variable_scope('zsc_preprocessing'):
            red, green, blue = tf.split(self.input_image, num_or_size_splits=3, axis=3)
            x = tf.concat([
                           tf.subtract(blue/255.0, 0.5)/0.5,
                           tf.subtract(green/255.0, 0.5)/0.5,
                           tf.subtract(red/255.0, 0.5)/0.5,
                          ], 3)
            
        with tf.variable_scope('zsc_feature'):
            #none,none,none,3
            x = PrimaryConv('PrimaryConv',x,32,self.norm,self.activate,self.is_training)
            skip_0 = x
            #none,none/2,none/2,32
            
            index = '1'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_1 = x
            #none,none/2,none/2,16
            
            index = '2'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_2 = x
            #none,none/4,none/4,24
            
            index = '3'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_3 = x
            #none,none/4,none/4,24
            
            index = '4'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_4 = x
            #none,none/8,none/8,32
            
            index = '5'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_5 = x
            #none,none/8,none/8,32
            
            index = '6'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_6 = x
            #none,none/16,none/16,64
            
            index = '7'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_7 = x
            #none,none/16,none/16,64
            
            index = '8'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_8 = x
            #none,none/16,none/16,96
            
            index = '9'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_9 = x
            #none,none/16,none/16,160
            
            index = '10'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_10 = x
            #none,none/32,none/32,160
            
            index = '11'
            x = Mobilenetv2Block('Mobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_11 = x
            #none,none/32,none/32,320
            
            
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('zsc_classifier'):
                
                global_pool = tf.reduce_mean(x,[1,2],keep_dims=True)
                
                self.classifier_logits = tf.reshape(_conv_block('Logits',global_pool,self.num_classes,1,1,'SAME',self.norm,None,self.is_training),
                                                    [tf.shape(global_pool)[0],self.num_classes])
        elif Detection_or_Classifier=='detection':
            with tf.variable_scope('zsc_detection'):
                with tf.variable_scope('zsc_pred'):
                    self.pred1 = _conv_block('pred_1',skip_3,self.ssd_default_box_size[0]*(1+self.num_classes+4),3,1,'SAME',None,None,self.is_training)
                    num_0123 = tf.shape(self.pred1)
                    self.pred1 = tf.reshape(self.pred1,[-1,num_0123[1]*num_0123[2]*self.ssd_default_box_size[0],1+self.num_classes+4])
                    
                    self.pred2 = _conv_block('pred_2',skip_5,self.ssd_default_box_size[1]*(1+self.num_classes+4),3,1,'SAME',None,None,self.is_training)
                    num_0123 = tf.shape(self.pred2)
                    self.pred2 = tf.reshape(self.pred2,[-1,num_0123[1]*num_0123[2]*self.ssd_default_box_size[1],1+self.num_classes+4])
                    
                    self.pred3 = _conv_block('pred_3',skip_7,self.ssd_default_box_size[2]*(1+self.num_classes+4),3,1,'SAME',None,None,self.is_training)
                    num_0123 = tf.shape(self.pred3)
                    self.pred3 = tf.reshape(self.pred3,[-1,num_0123[1]*num_0123[2]*self.ssd_default_box_size[2],1+self.num_classes+4])
                    
                    self.pred4 = _conv_block('pred_4',skip_8,self.ssd_default_box_size[3]*(1+self.num_classes+4),3,1,'SAME',None,None,self.is_training)
                    num_0123 = tf.shape(self.pred4)
                    self.pred4 = tf.reshape(self.pred4,[-1,num_0123[1]*num_0123[2]*self.ssd_default_box_size[3],1+self.num_classes+4])
                    
                    self.pred5 = _conv_block('pred_5',skip_10,self.ssd_default_box_size[4]*(1+self.num_classes+4),3,1,'SAME',None,None,self.is_training)
                    num_0123 = tf.shape(self.pred5)
                    self.pred5 = tf.reshape(self.pred5,[-1,num_0123[1]*num_0123[2]*self.ssd_default_box_size[4],1+self.num_classes+4])
                    
                    self.pred6 = _conv_block('pred_6',skip_11,self.ssd_default_box_size[5]*(1+self.num_classes+4),3,1,'SAME',None,None,self.is_training)
                    num_0123 = tf.shape(self.pred6)
                    self.pred6 = tf.reshape(self.pred6,[-1,num_0123[1]*num_0123[2]*self.ssd_default_box_size[5],1+self.num_classes+4])
                    
                    self.pred = tf.concat([self.pred1,self.pred2,self.pred3,self.pred4,self.pred5,self.pred6],axis=1)
                    
        self.__init__output()
        
        if Detection_or_Classifier=='classifier':
            pass
        elif Detection_or_Classifier=='detection':
            self.__prob()
    def __prob(self):
        #only for one image
        def correct_boxes(infos):
            pred_min_xy = infos[:,:2]*self.original_wh#框数,2
            pred_max_xy = infos[:,2:4]*self.original_wh#框数,2
            
            zeros = tf.zeros_like(pred_min_xy)
            ones = self.original_wh*tf.ones_like(pred_min_xy)
            pred_min_xy = tf.where(pred_min_xy>zeros,pred_min_xy,zeros)
            pred_min_xy = tf.where(pred_min_xy<ones,pred_min_xy,ones)
            pred_max_xy = tf.where(pred_max_xy>zeros,pred_max_xy,zeros)
            pred_max_xy = tf.where(pred_max_xy<ones,pred_max_xy,ones)
            return tf.concat([pred_min_xy,pred_max_xy,infos[:,4:]],axis=-1,name='zsc_output')
        def nms(infos,nms_threshold=0.4):
            #提取batch
            #infos_mask = tf.ones_like(infos)#框数,4+1+self.num_classes
            #batch = tf.cast(tf.reduce_sum(infos_mask)/tf.reduce_sum(infos_mask,axis=1)[0],tf.int32)
            batch = tf.shape(infos)[0]
            
            #先把infos按照最大class概率重排序
            #pred_max_class = tf.reduce_max(infos[:,5:],axis=1)#batch,
            #ix = tf.nn.top_k(tf.transpose(tf.expand_dims(pred_max_class,axis=1),[1,0]), batch, sorted=True, name="top_anchors").indices#1,batch
            #infos = tf.gather_nd(infos,tf.transpose(ix,[1,0]))#batch,4+1+self.num_classes
            
            pred_min_yx = infos[:,1::-1]#batch,2
            pred_max_yx = infos[:,3:1:-1]#batch,2
            pred_yx = tf.concat([pred_min_yx,pred_max_yx],axis=-1)#batch,4
            pred_max_class = tf.reduce_max(infos[:,4:],axis=1)#batch,
            indices = tf.image.non_max_suppression(pred_yx, pred_max_class, batch,nms_threshold, name="non_max_suppression")
            infos = tf.gather(infos, indices,name='zsc_output')
            return infos

        pred_confidence = self.pred[:,:,0]
        pred_class = self.pred[:,:,1:-4]
        pred_location = self.pred[:,:,-4:]
        
        positive = tf.where( tf.greater(pred_confidence,0.6) )
        pred_class = tf.gather_nd(pred_class,positive)#batch,num_classes
        pred_location = tf.gather_nd(pred_location,positive)#batch,4
        
        num = tf.shape(pred_class)[0]
        pred_class = tf.nn.softmax(pred_class,-1)
        pred_max_class = tf.reduce_max(pred_class,axis=-1)
        
        #ix = tf.nn.top_k(tf.transpose(tf.expand_dims(pred_max_class,axis=1),[1,0]), tf.floordiv(num,100), sorted=True, name="top_anchors").indices
        #pred_class = tf.gather_nd(pred_class,tf.transpose(ix,[1,0]))
        #pred_location = tf.gather_nd(pred_location,tf.transpose(ix,[1,0]))
        #positive = tf.where( tf.greater(pred_max_class,0.6) )
        #pred_class = tf.gather_nd(pred_class,positive)#batch,num_classes
        #pred_location = tf.gather_nd(pred_location,positive)#batch,4

        location_center = pred_location[:,:2]
        location_half_wh = tf.truediv(pred_location[:,2:],2.0)
        location_mins = tf.subtract(location_center,location_half_wh)
        location_maxs = tf.add(location_center,location_half_wh)
        
        self.infos = tf.concat([location_mins,location_maxs,pred_class],axis=-1)
        self.infos = correct_boxes(self.infos)
        self.infos = nms(self.infos,0.4)
    def do_nms(self,boxes, nms_thresh):
        def _interval_overlap(interval_a, interval_b):
            x1, x2 = interval_a
            x3, x4 = interval_b

            if x3 < x1:
                if x4 < x1:
                    return 0
                else:
                    return min(x2,x4) - x1
            else:
                if x2 < x3:
                    return 0
                else:
                    return min(x2,x4) - x3 
        def bbox_iou(box1, box2):
            intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
            intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    
            intersect = intersect_w * intersect_h

            w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
            w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    
            union = w1*h1 + w2*h2 - intersect
            print(intersect,union)
            return float(intersect) / union+1e-4
        for c in range(self.num_classes-1):
            sorted_indices = np.argsort([-box[4:][c] for box in boxes])

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i][4:][c] == 0: continue
             
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j][4:][c] = 0
        return boxes
    def __init__output(self):
        with tf.variable_scope('output'):
            regularzation_loss = self.loss.regularzation_loss()
            
            if Detection_or_Classifier=='classifier':
                self.all_loss = self.loss.sparse_softmax_loss(self.classifier_logits,self.y)
                self.all_loss += regularzation_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=1,decay_rate=0.98)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate)
                    self.train_op = self.optimizer.minimize(self.all_loss,global_step=self.global_epoch_tensor)
                
                self.y_out_softmax = tf.nn.softmax(self.classifier_logits,name='zsc_output')
                
                self.y_out_argmax = tf.cast(tf.argmax(self.y_out_softmax, axis=-1),tf.int32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))
                
                self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.y_out_softmax,self.y,5),tf.float32))

                #with tf.name_scope('train-summary-per-iteration'):
                #    tf.summary.scalar('loss', self.all_loss)
                #    tf.summary.scalar('acc', self.accuracy)
                #    tf.summary.scalar('acc', self.accuracy_top_5)
                #    self.summaries_merged = tf.summary.merge_all()
            elif Detection_or_Classifier=='detection':
                self.loss_class,self.loss_location,self.loss_confidence,self.loss_unconfidence = self.loss.ssd_loss(self.num_classes,self.pred,self.ground_truth,self.positive,self.negative,True)
                self.all_loss = tf.reduce_mean(self.loss_class+self.loss_location+self.loss_confidence+self.loss_unconfidence)+regularzation_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=1,decay_rate=0.98)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate)
                    self.train_op = self.optimizer.minimize(self.all_loss,global_step=self.global_epoch_tensor)
                
                #with tf.name_scope('train-summary-per-iteration'):
                #    tf.summary.scalar('loss', self.all_loss)
                #    tf.summary.scalar('class loss', tf.reduce_mean(self.loss_class))
                #    tf.summary.scalar('class location', tf.reduce_mean(self.loss_location))
                #    self.summaries_merged = tf.summary.merge_all()
    def __init_input(self):
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('input'):
                self.input_image = tf.placeholder(tf.float32,[None,None,None,3],name='zsc_input')#训练、测试用
                self.y = tf.placeholder(tf.int32, [None],name='zsc_input_target')#训练、测试用
                self.is_training = tf.placeholder(tf.float32,name='zsc_is_train')#训练、测试用
                self.is_training = tf.equal(self.is_training,1.0)
        elif Detection_or_Classifier=='detection':
            with tf.variable_scope('input'):
                self.input_image = tf.placeholder(tf.float32,[None,None,None,3],name='zsc_input')#训练、测试用
                self.original_wh = tf.placeholder(tf.float32,[None,2],name='zsc_original_wh')#仅测试用
                self.is_training = tf.placeholder(tf.float32,name='zsc_is_train')#训练、测试（不一定）用
                self.is_training = tf.equal(self.is_training,1.0)
                self.ground_truth = tf.placeholder(tf.float32,[None,None,1+4],name='ground_truth')
                self.positive = tf.placeholder(tf.float32,[None,None],name='positive')
                self.negative = tf.placeholder(tf.float32,[None,None],name='negative')
    def __init_global_epoch(self):
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
    def __init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

##LOSS
class Loss():
    def __init__(self):
        pass
    #regularzation loss
    def regularzation_loss(self):
        return sum(tf.get_collection("regularzation_loss"))
    
    #sparse softmax loss
    def sparse_softmax_loss(self, logits, labels):
        labels = tf.to_int32(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
            logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    #focal loss
    def focal_loss(self, prediction_tensor, target_tensor, alpha=0.25, gamma=2):
        #prediction_tensor [batch,num_anchors,num_classes]
        #target_tensor     [batch,num_anchors,num_classes]
        sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
        
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent,2)
    
    #smooth_L1
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x),1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))
    
    def ssd_loss(self, num_classes, pred, ground_truth, positive, negative, use_focal_loss=True):
        #pred [batch,num_anchors,num_classes+4]
        #ground_truth [batch,num_anchors,1+4]
        #positive [batch,num_anchors]
        #negative [batch,num_anchors]
        ground_truth_count = tf.add(positive,negative)
        if use_focal_loss:
            loss_class = self.focal_loss(pred[:,:,1:-4],tf.one_hot(tf.cast(ground_truth[:,:,0],tf.int32),num_classes))
        else:
            loss_class = self.sparse_softmax_loss(pred[:,:,1:-4],tf.cast(ground_truth[:,:,0],tf.int32))
        self.loss_location = tf.truediv(
                                        tf.reduce_sum(
                                                      tf.multiply(
                                                                  tf.reduce_sum(
                                                                                self.smooth_L1(
                                                                                               tf.subtract(
                                                                                                           ground_truth[:,:,1:], 
                                                                                                           pred[:,:,-4:]
                                                                                                           )
                                                                                               ),
                                                                                2
                                                                                ), 
                                                                  positive
                                                                  ),
                                                      1), 
                                        tf.reduce_sum(positive,1)
                                        )
        self.loss_class = tf.truediv(
                                     tf.reduce_sum(
                                                   tf.multiply(
                                                               loss_class,
                                                               ground_truth_count),
                                                   1), 
                                     tf.reduce_sum(ground_truth_count,1)
                                     )
        self.loss_confidence = tf.truediv(
                                        tf.reduce_sum(
                                                      tf.multiply(
                                                                 self.smooth_L1(
                                                                                tf.subtract(
                                                                                            positive, 
                                                                                            pred[:,:,0]
                                                                                            )
                                                                                ),
                                                                  positive
                                                                  ),
                                                      1), 
                                        tf.reduce_sum(positive,1)
                                        )
        self.loss_unconfidence = tf.truediv(
                                            tf.reduce_sum(
                                                          tf.multiply(
                                                                     self.smooth_L1(
                                                                                    tf.subtract(
                                                                                                negative, 
                                                                                                pred[:,:,0]
                                                                                                )
                                                                                    ),
                                                                      negative
                                                                      ),
                                                          1), 
                                            tf.reduce_sum(ground_truth_count,1)
                                            )
        return self.loss_class,self.loss_location,self.loss_confidence,self.loss_unconfidence
################################################################################################################
################################################################################################################
################################################################################################################
##Mobilenetv2Block
def Mobilenetv2Block(name,x,ratio=6,num_filters=16,repeat=1,stride=1,use_depthwise=True,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        if stride==1:
            for i in range(repeat):
                x = DepthwiseBlock('depthwiseblock_{}'.format(i),x,ratio,num_filters,use_depthwise,norm,activate,is_training)
        else:
            x = Transition('Transition',x,ratio,num_filters,use_depthwise,norm,activate,is_training)
        
        return x
def DepthwiseBlock(name,x,ratio,num_filters=16,use_depthwise=True,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        if use_depthwise:
            input = x
            
            x = _conv_block('conv_0',x,ratio*num_filters,1,1,'SAME',norm,activate,is_training)
            x = _depthwise_conv2d('depthwise',x,1,3,1,'SAME',norm,activate,is_training)
            x = _conv_block('conv_1',x,num_filters,1,1,'SAME',norm,activate,is_training)

            x = x*SE('attention',x,norm,activate,is_training)
            
            if input.get_shape().as_list()[-1]==x.get_shape().as_list()[-1]:
                pass
            else:
                input = _conv_block('conv_2',input,num_filters,1,1,'SAME',norm,activate,is_training)
            x += input
        else:
            x = _conv_block('conv',x,num_filters,1,1,'SAME',norm,activate,is_training)
        return x
def Transition(name,x,ratio=6,num_filters=16,use_depthwise=True,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        if use_depthwise:
            x = _conv_block('conv_0',x,ratio*num_filters,1,1,'SAME',norm,activate,is_training)
            x = _depthwise_conv2d('depthwise',x,1,3,2,'SAME',norm,activate,is_training)
            x = _conv_block('conv_1',x,num_filters,1,1,'SAME',norm,activate,is_training)
        else:
            x = _conv_block('conv',x,num_filters,3,2,'SAME',norm,activate,is_training)
        
        return x
##primary_conv
def PrimaryConv(name,x,num_filters=32,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        #none,none,none,3
        x = _conv_block('conv_0',x,num_filters,3,2,'SAME',norm,activate,is_training)#none,none/2,none/2,num_filters
        return x
##_conv_block
def _conv_block(name,x,num_filters=16,kernel_size=3,stride=2,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],num_filters])
        x = tf.nn.conv2d(x,w,[1,stride,stride,1],padding=padding,name='conv')
        
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            b = tf.get_variable('bias',num_filters,tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass

        return x
##_depthwise_conv2d
def _depthwise_conv2d(name,x,scale=1,kernel_size=3,stride=1,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name) as scope:
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],scale])
        x = tf.nn.depthwise_conv2d(x, w, [1,stride,stride,1], padding)
        
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            b = tf.get_variable('bias',scale,tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass
        return x
##_group_conv with channel shuffle use depthwise_conv2d
def _group_conv(name,x,group=4,num_filters=16,kernel_size=1,stride=1,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        C = x.shape.as_list()[-1]
        num_012 = tf.shape(x)[:3]
        assert C%group==0 and num_filters%group==0
        
        w = GetWeight('weight',[kernel_size,kernel_size,C,num_filters//group])
        x = tf.nn.depthwise_conv2d(x, w, [1,stride,stride,1], padding)
        
        x = tf.reshape(x,tf.concat([ [num_012[0]], tf.cast(num_012[1:3]/kernel_size,tf.int32), tf.cast([group, C//group, num_filters//group],tf.int32)],axis=-1))
        x = tf.reduce_sum(x,axis=4)
        x = tf.transpose(x,[0,1,2,4,3])
        x = tf.reshape(x,tf.concat([ [num_012[0]], tf.cast(num_012[1:3]/kernel_size,tf.int32), tf.cast([num_filters],tf.int32)],axis=-1))
        
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            pass
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass
        
        return x
##senet
def SE(name,x,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        #none,none,none,C
        C = x.get_shape().as_list()[-1]
        #SEnet channel attention
        weight_c = tf.reduce_mean(x,[1,2],keep_dims=True)#none,1,1,C
        weight_c = _conv_block('conv_1',weight_c,C//16,1,1,'SAME',None,activate,is_training)
        weight_c = _conv_block('conv_2',weight_c,C,1,1,'SAME',None,None,is_training)
        
        weight_c = tf.nn.sigmoid(weight_c)#none,1,1,C
        
        return weight_c
##weight variable
def GetWeight(name,shape,weights_decay = 0.00004):
    with tf.variable_scope(name):
        #w = tf.get_variable('weight',shape,tf.float32,initializer=VarianceScaling())
        w = tf.get_variable('weight',shape,tf.float32,initializer=glorot_uniform_initializer())
        weight_decay = tf.multiply(tf.nn.l2_loss(w), weights_decay, name='weight_loss')
        tf.add_to_collection('regularzation_loss', weight_decay)
        return w
##initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math
def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="uniform",
                          seed=seed,
                          dtype=dtype)
def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="normal",
                          seed=seed,
                          dtype=dtype)
def _compute_fans(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out
class VarianceScaling():
    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None,
                 dtype=dtypes.float32):
      if scale <= 0.:
          raise ValueError("`scale` must be positive float.")
      if mode not in {"fan_in", "fan_out", "fan_avg"}:
          raise ValueError("Invalid `mode` argument:", mode)
      distribution = distribution.lower()
      if distribution not in {"normal", "uniform"}:
          raise ValueError("Invalid `distribution` argument:", distribution)
      self.scale = scale
      self.mode = mode
      self.distribution = distribution
      self.seed = seed
      self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
      if dtype is None:
          dtype = self.dtype
      scale = self.scale
      scale_shape = shape
      if partition_info is not None:
          scale_shape = partition_info.full_shape
      fan_in, fan_out = _compute_fans(scale_shape)
      if self.mode == "fan_in":
          scale /= max(1., fan_in)
      elif self.mode == "fan_out":
          scale /= max(1., fan_out)
      else:
          scale /= max(1., (fan_in + fan_out) / 2.)
      if self.distribution == "normal":
          stddev = math.sqrt(scale)
          return random_ops.truncated_normal(shape, 0.0, stddev,
                                             dtype, seed=self.seed)
      else:
          limit = math.sqrt(3.0 * scale)
          return random_ops.random_uniform(shape, -limit, limit,
                                           dtype, seed=self.seed)
##LeakyRelu
def LeakyRelu(x, leak=0.1, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
##selu
def selu(x,name='selu'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
##swish
def swish(x,name='swish'):
    with tf.variable_scope(name):
        beta = tf.Variable(1.0,trainable=True)
        return x*tf.nn.sigmoid(beta*x)
##crelu 注意使用时深度要减半
def crelu(x,name='crelu'):
    with tf.variable_scope(name):
        x = tf.concat([x,-x],axis=-1)
        return tf.nn.relu(x)
def prelu(inputs,name='prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs-abs(inputs))*0.5
        return pos + neg
################################################################################################################
################################################################################################################
################################################################################################################

if __name__=='__main__':
    import time
    from functools import reduce
    from operator import mul
    
    import numpy as np

    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    model = Mobilenetv2(num_classes=23)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
        
        
        feed_dict={model.input_image:np.random.randn(1,224,224,3),
                   model.is_training:1.0,
                   model.ground_truth:np.concatenate([np.ones((1,26460,1)),np.random.randn(1,26460,4)],axis=-1),
                   model.positive:np.ones((2,26460)),
                   model.negative:np.zeros((2,26460)),
                   model.original_wh:[[256,256]]}
        
        start = time.time()
        out = sess.run(model.infos,feed_dict=feed_dict)
        print('Spend Time:{}'.format(time.time()-start))
        
        print(out)
