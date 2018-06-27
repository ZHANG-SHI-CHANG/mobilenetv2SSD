﻿import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np
import cv2

from Mobilenetv2 import Mobilenetv2 as Model
from Mobilenetv2 import Detection_or_Classifier

from detection_dataloader import create_training_instances,BatchGenerator
from classifier_dataloader import DataLoader

import os
import glob
import platform
if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

from functools import reduce
from operator import mul
def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

num_epochs = 10000
max_to_keep = 2
save_model_every = 3
test_every = 1

is_train = False

##############################################
batch_size = 4
##############################################

def normalize(image):
    return image/255.

def main():
    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)

    # Train class
    trainer = Train(sess)

    if is_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()
    else:
        print("Testing...")
        trainer.test()
        print("Testing Finished\n\n")

class Train:
    """Trainer class for the CNN.
    It's also responsible for loading/saving the model checkpoints from/to experiments/experiment_name/checkpoint_dir"""

    def __init__(self, sess):
        self.sess = sess
        self.dataset_root = 'F:\\Learning\\tensorflow\\detect\\Dataset\\'
        
        if Detection_or_Classifier=='classifier':
            self.labels = list(range(18))
            
            self.train_data = DataLoader(root=os.path.join(self.dataset_root,'data','train'),classes=len(self.labels),batch=batch_size)
            
            print("Building the model...")
            self.model = Model(num_classes=len(self.labels))
            print("Model is built successfully\n\n")
            
        elif Detection_or_Classifier=='detection':
            train_ints, valid_ints, self.labels = create_training_instances(
            os.path.join(self.dataset_root,'Fish','Annotations'),
            os.path.join(self.dataset_root,'Fish','JPEGImages'),
            'data.pkl',
            '','','',False,
            ['heidiao','niyu','lvqimamiantun','hualu','heijun','dalongliuxian','tiaoshiban']
            #['person','head','hand','foot','aeroplane','tvmonitor','train','boat','dog','chair',
            # 'bird','bicycle','bottle','sheep','diningtable','horse','motorbike','sofa','cow',
            # 'car','cat','bus','pottedplant']
            )
            
            self.train_data = BatchGenerator(train_ints,self.labels,batch_size)
            
            print("Building the model...")
            self.model = Model(num_classes=23)
            print("Model is built successfully\n\n")
        
        #tf.profiler.profile(tf.get_default_graph(),options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter(), cmd='scope')
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
        
        self.use_classifier_pretrain = True
        if not self.use_classifier_pretrain:
            var = tf.global_variables()
        else:
            var = tf.trainable_variables()
        var_list = [val for val in var]
        if Detection_or_Classifier=='detection' and self.use_classifier_pretrain:
            var_list = [val for val in var if 'zsc_detection' not in val.name]
        
        if Detection_or_Classifier=='classifier':
            var = tf.global_variables()
            var_list = [val for val in var if 'Logits/bias' not in val.name]
        self.saver = tf.train.Saver(var_list=var_list,max_to_keep=max_to_keep)
        
        self.save_checkpoints_path = os.path.join(os.getcwd(),'checkpoints',Detection_or_Classifier)
        if not os.path.exists(self.save_checkpoints_path):
            os.makedirs(self.save_checkpoints_path)

        # Initializing the model
        self.init = None
        self.__init_model()

        # Loading the model checkpoint if exists
        self.__load_model()
        
        #summary_dir = os.path.join(os.getcwd(),'logs',Detection_or_Classifier)
        #if not os.path.exists(summary_dir):
        #    os.makedirs(summary_dir)
        #summary_dir_train = os.path.join(summary_dir,'train')
        #if not os.path.exists(summary_dir_train):
        #    os.makedirs(summary_dir_train)
        #summary_dir_test = os.path.join(summary_dir,'test')
        #if not os.path.exists(summary_dir_test):
        #    os.makedirs(summary_dir_test)
        #self.train_writer = tf.summary.FileWriter(summary_dir_train,sess.graph)
        #self.test_writer = tf.summary.FileWriter(summary_dir_test)

    ############################################################################################################
    # Model related methods
    def __init_model(self):
        print("Initializing the model...")
        self.init = tf.group(tf.global_variables_initializer())
        self.sess.run(self.init)
        print("Model initialized\n\n")

    def save_model(self):
        var = tf.global_variables()
        var_list = [val for val in var]
        self.saver = tf.train.Saver(var_list=var_list,max_to_keep=max_to_keep)
        
        print("Saving a checkpoint")
        self.saver.save(self.sess, self.save_checkpoints_path+'/'+Detection_or_Classifier, self.model.global_step_tensor)
        print("Checkpoint Saved\n\n")
        
        print('Saving a pb')
        if Detection_or_Classifier=='classifier':
            output_graph_def = graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(), ['output/zsc_output'])
            #tflite_model = tf.contrib.lite.toco_convert(output_graph_def, [self.model.input_image], [self.model.y_out_softmax])
            #open(Detection_or_Classifier+".tflite", "wb").write(tflite_model)
        elif Detection_or_Classifier=='detection':
            output_graph_def = graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(), ['zsc_output'])
        tf.train.write_graph(output_graph_def, self.save_checkpoints_path, Detection_or_Classifier+'.pb', as_text=False)
        print('pb saved\n\n')
        
    def __load_model(self):
        if Detection_or_Classifier=='detection' and self.use_classifier_pretrain:
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(os.getcwd(),'checkpoints','classifier'))
            if latest_checkpoint:
                print("loading classifier checkpoint {} ...\n".format(latest_checkpoint))
                self.saver.restore(self.sess, latest_checkpoint)
                print("classifier model success loaded\n\n")
            else:
                print('loading classifier model failure!!')
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self.save_checkpoints_path)
            if latest_checkpoint:
                print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                self.saver.restore(self.sess, latest_checkpoint)
                print("Checkpoint loaded\n\n")
            else:
                print("First time to train!\n\n")
    ############################################################################################################
    # Train and Test methods
    def train(self):
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, num_epochs + 1, 1):

            batch = 0
            
            loss_list = []
            
            if Detection_or_Classifier=='classifier':
                acc_list = []
                acc_5_list = []
                for X_batch, y_batch in self.train_data.next():
                    print('Training epoch:{},batch:{}\n'.format(cur_epoch,batch))
                    
                    cur_step = self.model.global_step_tensor.eval(self.sess)
                    
                    feed_dict = {self.model.input_image: X_batch,
                                 self.model.y: y_batch,
                                 self.model.is_training: 1.0
                                 }
                                 
                    #_, loss, acc,acc_5, summaries_merged = self.sess.run(
                    #[self.model.train_op, self.model.all_loss, self.model.accuracy,self.model.accuracy_top_5, self.model.summaries_merged],
                    #feed_dict=feed_dict)
                    _, loss, acc,acc_5 = self.sess.run(
                    [self.model.train_op, self.model.all_loss, self.model.accuracy,self.model.accuracy_top_5],
                    feed_dict=feed_dict)
                    
                    print('loss:' + str(loss)+'|'+'accuracy:'+str(acc)+'|'+'top_5:'+str(acc_5))
                    
                    loss_list += [loss]
                    acc_list += [acc]
                    acc_5_list += [acc_5]
                    
                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_step + 1})

                    #self.train_writer.add_summary(summaries_merged,cur_step)

                    if batch > self.train_data.__len__():
                        batch = 0
                    
                        avg_loss = np.mean(loss_list).astype(np.float32)
                        avg_accuracy = np.mean(acc_list).astype(np.float32)
                        avg_top5 = np.mean(acc_5_list).astype(np.float32)
                        
                        self.model.global_epoch_assign_op.eval(session=self.sess,
                                                               feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                        print("\nEpoch-" + str(cur_epoch) + '|' + 'avg loss:' + str(avg_loss)+'|'+'avg accuracy:'+str(avg_accuracy)+'|'+'avg top_5:'+str(avg_top5)+'\n')
                        break
                    
                    if batch==0 and cur_epoch%99==0:
                        #opts = tf.profiler.ProfileOptionBuilder.float_operation()    
                        #flops = tf.profiler.profile(tf.get_default_graph(), run_meta=tf.RunMetadata(), cmd='op', options=opts)
                        #if flops is not None:
                        #    print('flops:{}'.format(flops.total_float_ops))
                        '''
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        
                        _,summaries_merged = self.sess.run([self.model.train_op, self.model.summaries_merged],
                                                       feed_dict=feed_dict,
                                                       options=run_options,
                                                       run_metadata=run_metadata)
                        
                        self.train_writer.add_run_metadata(run_metadata, 'epoch{}batch{}'.format(cur_epoch,cur_step))
                        self.train_writer.add_summary(summaries_merged, cur_step)
                        '''
                        pass

                    if batch == 400:
                        self.save_model()

                    batch += 1
                
                if cur_epoch % save_model_every == 0 and cur_epoch != 0:
                    self.save_model()
                    
            elif Detection_or_Classifier=='detection':
                class_loss_list = []
                location_loss_list = []
                for x_batch, ground_truth, positive, negative in self.train_data.next():
                    print('Training epoch:{},batch:{}\n'.format(cur_epoch,batch))
                    
                    cur_step = self.model.global_step_tensor.eval(self.sess)
                    
                    print('hide image')
                    S_ratio = 32
                    S = x_batch.shape[1]/S_ratio
                    for _batch in range(x_batch.shape[0]):
                        IndexList = []
                        for _ in range(S_ratio*S_ratio):
                            IndexList.append(np.random.randint(0,2))
                        IndexList = np.array(IndexList).reshape(S_ratio,S_ratio).tolist()
                        for i in range(S_ratio):
                            for j in range(S_ratio):
                                if IndexList[i][j]==1:
                                    pass
                                else:
                                    x_batch[_batch,int(S*i):int(S*(i+1)),int(S*j):int(S*(j+1)),:] = 0.0
                    print('hide success')
                    
                    feed_dict={self.model.input_image:x_batch,
                               self.model.is_training:1.0,
                               self.model.ground_truth:ground_truth,
                               self.model.positive:positive,
                               self.model.negative:negative}
                    
                    #_, loss, class_loss, location_loss, summaries_merged = self.sess.run(
                    #    [self.model.train_op, self.model.all_loss, self.model.loss_class, self.model.loss_location,self.model.summaries_merged],
                    #    feed_dict=feed_dict)
                    _, loss, class_loss, location_loss = self.sess.run(
                        [self.model.train_op, self.model.all_loss, self.model.loss_class, self.model.loss_location],
                        feed_dict=feed_dict)
                    
                    print('Batch:{} class_loss:{} location_loss:{} all_loss:{}'.format(batch,np.mean(class_loss),np.mean(location_loss),loss))
                    loss_list += [loss]
                    class_loss_list += [class_loss]
                    location_loss_list += [location_loss]

                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_step + 1})

                    #self.train_writer.add_summary(summaries_merged,cur_step)

                    if batch*batch_size > self.train_data.__len__():
                        batch = 0
                    
                        avg_loss = np.mean(loss_list).astype(np.float32)
                        avg_class_loss = np.mean(class_loss_list).astype(np.float32)
                        avg_location_loss = np.mean(location_loss_list).astype(np.float32)
                        
                        self.model.global_epoch_assign_op.eval(session=self.sess,
                                                               feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                        print('Epoch-{} | avg_class_loss:{} | avg_location_loss:{} | avg_all_loss:{}'.format(cur_epoch,avg_class_loss,avg_location_loss,avg_loss))
                        break
                    
                    if batch==0 and cur_epoch%99==0:
                        #opts = tf.profiler.ProfileOptionBuilder.float_operation()    
                        #flops = tf.profiler.profile(tf.get_default_graph(), run_meta=tf.RunMetadata(), cmd='op', options=opts)
                        #if flops is not None:
                        #    print('flops:{}'.format(flops.total_float_ops))
                        '''
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        
                        _,summaries_merged = self.sess.run([self.model.train_op, self.model.summaries_merged],
                                                       feed_dict=feed_dict,
                                                       options=run_options,
                                                       run_metadata=run_metadata)
                        
                        self.train_writer.add_run_metadata(run_metadata, 'epoch{}batch{}'.format(cur_epoch,cur_step))
                        self.train_writer.add_summary(summaries_merged, cur_step)
                        '''
                        pass

                    batch += 1
                
                if cur_epoch % save_model_every == 0 and cur_epoch != 0:
                    self.save_model()
                
                if cur_epoch % test_every == 0:
                    print('start test')
                    self.test()
                    print('end test')
    def test(self):
        if Detection_or_Classifier=='classifier':
            
                ImageList = []
                ImageLabelList = []
                with open(os.path.join(os.getcwd(),'test_images',Detection_or_Classifier,'imagelist.txt'),'r') as f:
                    for i,line in enumerate(f.readlines()):
                        ImageList.append( cv2.resize(cv2.imread(os.path.join(os.getcwd(),'test_images',Detection_or_Classifier,line.split(' ')[0])),(224,224)) )
                        ImageLabelList.append(int(line.split(' ')[1]))
                
                loss_list = []
                acc_list = []
                top5_list = []
                for i in range(0,len(ImageList)//batch_size):
                    print('process batch:{}'.format(i))               
  
                    x = np.stack(ImageList[i*batch_size:(i+1)*batch_size],axis=0).astype(np.float32)
                    y = np.array(ImageLabelList[i*batch_size:(i+1)*batch_size],dtype=np.int32)

                    feed_dict = {self.model.input_image: x,
                                 self.model.y: y,
                                 self.model.is_training: 0.0
                                 }
                                 
                    loss, acc,acc_5 = self.sess.run(
                    [self.model.all_loss, self.model.accuracy,self.model.accuracy_top_5],
                    feed_dict=feed_dict)
                    
                    loss_list.append(loss)
                    acc_list.append(acc)
                    top5_list.append(acc_5)
                    
                print('test avg loss:' + str(np.mean(loss_list))+'|'+'avg accuracy:'+str(np.mean(acc_list))+'|'+'avg top_5:'+str(np.mean(top5_list)))
                
        elif Detection_or_Classifier=='detection':
            labels = self.labels
                
            if not os.path.exists(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier)):
                os.makedirs(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier))
            
            for image_path in glob.glob(os.path.join(os.getcwd(),'test_images',Detection_or_Classifier,'*.jpg')):
                image_name = image_path.split(SplitSym)[-1]
                print('processing image {}'.format(image_name))

                image = cv2.imread(image_path)
                image_h,image_w,_ = image.shape
                _image = cv2.resize(image,(32*11,32*11))[np.newaxis,:,:,::-1]
                infos = self.sess.run(self.model.infos,
                                          feed_dict={self.model.input_image:_image,
                                                     self.model.is_training:0.0,
                                                     self.model.original_wh:[[image_w,image_h]]
                                                     }
                                          )
                print(infos.shape)
                #infos = self.model.do_nms(infos,0.4)
                image = self.draw_boxes(image, infos.tolist(), labels)
                cv2.imwrite(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier,image_name),image)
    def draw_boxes(self,image, boxes, labels, obj_thresh=0.6):
        def _constrain(min_v, max_v, value):
            if value < min_v: return min_v
            if value > max_v: return max_v
            return value
        
        for box in boxes:
            label_str = ''
            
            max_idx = -1
            max_score = 0
            for i in range(len(labels)):
                if box[4:][i] > obj_thresh:
                    label_str += labels[i]

                    if box[4:][i]>max_score:
                        max_score = box[4:][i]
                        max_idx = i
                    print(labels[i] + ': ' + str(box[4:][i]*100) + '%')
            print(max_idx)
            if max_idx >= 0:
                cv2.rectangle(image, (int(_constrain(1,image.shape[1]-2,box[0])),int(_constrain(1,image.shape[0]-2,box[1]))), 
                                     (int(_constrain(1,image.shape[1]-2,box[2])),int(_constrain(1,image.shape[0]-2,box[3]))), (0,0,255), 3)
                cv2.putText(image, 
                            labels[max_idx] + ' ' + str(max(box[4:])),
                            (int(box[0]), int(box[1]) - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0,0,255), 2)
        return image

if __name__=='__main__':
    main()
