#age network combining cycleGAN
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
f=open('logemb.txt','w')

def main(args):
    #define three networks
    network_G = importlib.import_module(args.model_def)              #import G Network  
    network_D = importlib.import_module(args.discriminator_def)      #import D Network (connected with G)
    network_F = importlib.import_module(args.Net_def)                #F network(same with G)

    #model dir
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')      #model name (named by time)1
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):                                   # Create the log directory if it doesn't exist 1
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):                                 # Create the model directory if it doesn't exist 1 
        os.makedirs(model_dir)

    # Write arguments to a text file 1
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))    #mark arguments
        
    # Store some git revision info in a text file in the log directory 1
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = facenet.get_dataset(args.data_dir)            #get train dataset
    if args.filter_filename:                           #not used
        train_set = filter_dataset(train_set, os.path.expanduser(args.filter_filename), 
            args.filter_percentile, args.filter_min_nrof_images_per_class)
    nrof_classes = len(train_set)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)       #facenet model
        print('Pre-trained model: %s' % pretrained_model)
   
    # not used here
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)



    # copy from faceNet
    def get_image_paths_and_labels(dataset):
        image_paths_flat = []
        labels_flat = []
        for i in range(len(dataset)):
            image_paths_flat += dataset[i].image_paths
            #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')#++++++++++++++++++++++++++++++++
            #print(dataset[i].image_paths)#++++++++++++++++++++++++++++++++
            labels_flat += [i] * len(dataset[i].image_paths)#labels_flat=age??????????????????????????????????????
        return image_paths_flat, labels_flat
    
    
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        # Get a list of image paths and their labels
        image_list, label_list = get_image_paths_and_labels(train_set)   #[path, path, ..., path], [1,1,2,2,2,3,3,4,...,9]
        assert len(image_list)>0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list 
        #labels and images into queue
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        
        range_size = array_ops.shape(labels)[0]
        
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32) 
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        #placeholder
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')    #connected to validate
        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

        image_embeddings_placeholder = tf.placeholder(tf.float32, shape=(None,1,args.embedding_size), name='image_embs')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
        
        input_queue = data_flow_ops.FIFOQueue(capacity=100000, dtypes=[tf.string, tf.int64,tf.float32],
                                    shapes=[(1,), (1,),(1,args.embedding_size,)], shared_name=None, name=None)   #(image path, labels, embedding)
        
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder,image_embeddings_placeholder], name='enqueue_op')
        
        nrof_preprocess_threads = 4
        images_and_labels = []

        #preprocess some images to extend database

        for _ in range(nrof_preprocess_threads):
            filenames, label ,image_embeddings= input_queue.dequeue()    #get (image path, labels, embedding) from queue
            print('filenames.shape, label.shape, image_embeddings.shape:')
            print(filenames.shape,label.shape,image_embeddings.shape)
            #print(filenames[1,1])  #queueeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecheck
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)  #BGR, , Decode .PNG read an image from filename=image path
                
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:  #run here
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
               
                
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image)) #add into image set named images
            #print(len(images))                                                  #1?????
            images_and_labels.append([images, label,image_embeddings])

        #from queue into batch
        image_batch, label_batch ,embeddings_batch= tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), (),(128)], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        print('Shape of embeddings_batch:')
        print(embeddings_batch.shape)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        embeddings_batch = tf.identity(embeddings_batch, 'emb_batch')
        
        print('Total number of classes:%d' %nrof_classes)
        print('Length of image list:%d' % len(image_list))
        print('Building training graph ...')

        # Build the inference graph, from inception resnet.py
        prelogits, ep = network_G.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        prelogits2, _ = network_D.inference(prelogits,ep, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        prelogits3, ep2 = network_F.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits2, len(train_set), activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='DiscriminatorLogits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')     #G feature
        embeddings2 = tf.nn.l2_normalize(prelogits3, 1, 1e-10, name='embeddings2')    #F feature
        labels = tf.nn.l2_normalize(prelogits2, 1, 1e-10, name='labels')              #D decision
        print('embeddingsshape:',embeddings.shape,embeddings_batch.shape)

        emb_loss = args.beta * tf.reduce_mean(tf.square(embeddings - embeddings_batch)) #!!!!!!!!!!!!!!!!!!!!!!!emb_loss=pretrained feature-F feature
        #print('emb_loss')
        #print(np.sum(emb_loss))

        # Add center loss for D
        center_loss_all=0.0
        if args.center_loss_factor>0.0:
            prelogits_center_loss, _ = facenet.center_loss(prelogits2, label_batch, args.center_loss_alfa, nrof_classes)
            center_loss_all = prelogits_center_loss * args.center_loss_factor
            # Do not add the center loss to regularization
            #regularization_losses = tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch 1
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the reg losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        #********************************************consistency_loss************************************************************************
        consistency_loss_G = args.betaG * tf.reduce_mean(tf.square(embeddings - embeddings2))
        consistency_loss_F = args.betaF * tf.reduce_mean(tf.square(embeddings - embeddings2))  #different coefficients

        #*******************************************unsupervised id loss*********************************************************************
        #anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        anchor = embeddings
        positive = embeddings2
        negative = embeddings
        id_loss = args.betaI * identity_loss(anchor, positive, negative, args.alpha, args.batch_size)

        #************************************************************************************************************************************


        # losses of three networks
        total_loss_G = tf.add_n([-cross_entropy_mean] +[-center_loss_all]+ [emb_loss]+[consistency_loss_G]+ regularization_losses, name='total_loss_G')   # loss of G_network, to decrease the discriminant
                                                                                                                                        #  of age while not change embeddings
        total_loss_D = tf.add_n([cross_entropy_mean] +[center_loss_all]+ regularization_losses, name='total_loss_D') 
        total_loss_F = tf.add_n([id_loss]+ [consistency_loss_F]+ regularization_losses, name='total_loss_F') 


        #optimizer###########################################################################################################################
        optimizer=args.optimizer
        loss_averages_op1 = facenet._add_loss_summaries(total_loss_D)
        loss_averages_op2 = facenet._add_loss_summaries(total_loss_G)
        loss_averages_op3 = facenet._add_loss_summaries(total_loss_F)  #just add an op?

        learning_rate2=learning_rate
        learning_rate3=learning_rate
        if optimizer=='ADAGRAD':
            opt1 = tf.train.AdagradOptimizer(learning_rate)
            opt2 = tf.train.AdagradOptimizer(learning_rate2)
            opt3 = tf.train.AdagradOptimizer(learning_rate3)
        elif optimizer=='ADADELTA':
            opt1 = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
            opt2 = tf.train.AdadeltaOptimizer(learning_rate2, rho=0.9, epsilon=1e-6)
            opt3 = tf.train.AdadeltaOptimizer(learning_rate3, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt1 = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
            opt2 = tf.train.AdamOptimizer(learning_rate2, beta1=0.9, beta2=0.999, epsilon=0.1)
            opt3 = tf.train.AdadeltaOptimizer(learning_rate3, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt1 = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
            opt2 = tf.train.RMSPropOptimizer(learning_rate2, decay=0.9, momentum=0.9, epsilon=1.0)
            opt3 = tf.train.AdadeltaOptimizer(learning_rate3, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt1 = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            opt2 = tf.train.MomentumOptimizer(learning_rate2, 0.9, use_nesterov=True)
            opt3 = tf.train.AdadeltaOptimizer(learning_rate3, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm2')
        t_vars=tf.trainable_variables()

        varlog=open('tvar.txt','w')
        for var in t_vars:
            varlog.write(var.name+'\n')
        varlog.close()

        #variables
        d_vars=[var for var in t_vars if 'Discriminator' in var.name]
        g_vars=[var for var in t_vars if 'InceptionResnetV1' in var.name]    #Generator? how to 
        f_vars=[var for var in t_vars if 'InceptionResnetV2' in var.name]    #Generator? how to
        #true loss for age discriminator and false loss for generator
        dt_grads = opt1.compute_gradients(total_loss_D, d_vars)
        df_grads = opt2.compute_gradients(total_loss_G, g_vars)
        di_grads = opt3.compute_gradients(total_loss_F, f_vars)
        dt_train_op = opt1.apply_gradients(dt_grads, global_step=global_step)
        df_train_op = opt2.apply_gradients(df_grads, global_step=global_step)
        di_train_op = opt3.apply_gradients(di_grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)


        # Add histograms for gradients. copy from train2
        
        for grad, var in dt_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        for grad, var in df_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        for grad, var in di_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            args.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
        with tf.control_dependencies([dt_train_op,df_train_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3) # save all variables++++++++++++++++++++++++++++++++++++0
        saver_G = tf.train.Saver(g_vars, max_to_keep=3)   #only save G variables0
        saver_F = tf.train.Saver(f_vars, max_to_keep=3)   #only save G variables0

        # Build the summary operation based on the TF collection of Summaries. 1
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph. 1
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)


        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver_G.restore(sess, pretrained_model)

            print('Running pretrain embeddings')#see embs**********************************
            len_all=len(label_list)  #*****This is ok 397459
            #print('len_all=')
            #print(len_all)
            emb_list=np.zeros((len_all,args.embedding_size))
            print(len_all//(args.batch_size*args.epoch_size)+1)

            #obtain pretain feature
            for ii in range(0,len_all//(args.batch_size*args.epoch_size)+1):
                emb_list=first_train(args, sess, 0, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    image_embeddings_placeholder,emb_list,embeddings,label_batch,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss_D, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)


            for ii in range(0,len_all):
                #print('pretrain embeddings')
                #print(np.sum(emb_list[ii,:]),file=f)    #cannot be printed,why~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!
                print(emb_list[ii,:],file=f) 

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size//2
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    image_embeddings_placeholder,emb_list,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss_D, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file, anchor, positive, negative, id_loss, consistency_loss_G, emb_loss)

                # Save variables and the metagraph if it doesn't exist already 1
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW 1
                if args.lfw_dir:
                    evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
                        embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer)
    return model_dir


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset

def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      image_embeddings_placeholder,emb_list, learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file, anchor, positive, negative, id_loss, consistency_loss_G, emb_loss):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    emb_epoch=np.array(emb_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    img_emb_array=np.expand_dims(np.array(emb_epoch),1)
    #print(img_emb_array.shape)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,image_embeddings_placeholder:img_emb_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run([id_loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([id_loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %   #############################################################output###########################################
              (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))   #regularization_losses np.sum(reg_loss)
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step



def identity_loss(anchor, positive, negative, alpha, people_per_batch):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """

    #pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    pos_dist = 0
    for k0 in xrange (people_per_batch):
        for k3 in xrange (people_per_batch):
            pos_dist += tf.reduce_sum(tf.square(tf.subtract(anchor[k0], positive[k0])), 0)

    neg_dist = 0
    for k1 in xrange (people_per_batch):
        for k2 in xrange(people_per_batch):
            neg_dist += tf.reduce_sum(tf.square(tf.subtract(anchor[k1], negative[k2])), 0)
        
    #basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    #loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    basic_loss=tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    loss = tf.maximum(basic_loss, 0.0)
      
    return loss
  


def first_train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
      image_embeddings_placeholder,emb_list,embeddings,label_batch,
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0
    
    lr=0     #learning rate=0??????????????????????????????????????????????????????????????????????????????????????????????

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = index_epoch
    image_epoch = np.array(image_list)[index_epoch]
    emb_epoch=np.array(emb_list)[index_epoch]
    #print(index_epoch.shape,emb_epoch.shape)
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    img_emb_array=np.expand_dims(np.array(emb_epoch),1)
    #print(index_epoch.shape,emb_epoch.shape)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,image_embeddings_placeholder:img_emb_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:False, batch_size_placeholder:args.batch_size}
        step, emb_ori,index_ori = sess.run([ global_step, embeddings,label_batch], feed_dict=feed_dict) #???????????????????????????????????????
        duration = time.time() - start_time
        #print(index_ori)
        #print("emb_shape:",emb_ori.shape,'ind_shape:',index_ori.shape)
        for i in range(0,args.batch_size):
            emb_list[index_ori[i],:]=emb_ori[i,:]      #in while?????????????????????????????
        #print(emb_list.shape)
        print('Epoch: [%d][%d/%d]\tTime %.3f\t' %
              (epoch, batch_number+1, args.epoch_size, duration))
        batch_number += 1
        train_time += duration
    return emb_list



def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0,len(image_paths)),1)
    image_paths_array = np.expand_dims(np.array(image_paths),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    
    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame)*2
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))#+++++++++++++++++++++++++++++++++++++
    
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb


    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))



def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='/home/d201/Wuyboo/pyCharm/cycleGAN/Logs')   #save logs
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='/home/d201/Wuyboo/pyCharm/cycleGAN/models')   #save trained models#####
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)  #usage of GPU
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.', #default='/home/d201/Wuyboo/pyCharm/AgeGAN/20170511-185253/')
        default='/home/d201/Wuyboo/pyCharm/AgeGAN/20170511-185253/model-20170511-185253.ckpt-80000') ##########original FaceNet model###########
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='/home/d201/Wuyboo/pyCharm/Datasets/IMDB/')
        #default='/home/d201/Wuyboo/FaceNet/FaceDatabase/cropcasia')
        #default='/home/d201/Wuyboo/pyCharm/Datasets/IMDB/')  #train dataset#########################################################
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='inception_resnet')   #G part
    parser.add_argument('--Net_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='inception_resnet_v2')   #F part
    parser.add_argument('--discriminator_def', type=str,
						help='Discriminator model definition. Points to a module containing the definition of the inference graph.', default='discriminator')   #D part
    parser.add_argument('--max_nrof_epochs', type=int, help='Number of epochs to run.', default=3)             #10-1000#####
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=30)   ##33
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int, help='Number of batches per epoch.', default=1000)            ##3333
    parser.add_argument('--embedding_size', type=int, help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')  
    parser.add_argument('--random_flip',  help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',  help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,  help='Keep probability of dropout for the fully connected layer(s).', default=1.0) # change to around 0.5 when overfitting ##2222
    parser.add_argument('--weight_decay', type=float, help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,  help='Center loss factor.', default=0.1)           #######in (0,1)#######111111111
    parser.add_argument('--center_loss_alfa', type=float,  help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--beta', type=float, help='Embeddings update rate for transfer loss.', default=300)    #The weight of emb_loss!!!!!!!!
    parser.add_argument('--betaI', type=float, help='Embeddings update rate for id loss.', default=100)    #The weight of emb_loss!!!!!!!!
    parser.add_argument('--betaG', type=float, help='Embeddings update rate for consistency loss of G.', default=10)
    parser.add_argument('--betaF', type=float, help='Embeddings update rate for consistency loss of F.', default=100)
    parser.add_argument('--alpha', type=float, help='Margin in identity loss.', default=20)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', '3ADADELTA', 'ADAM', 'RMSPROP', 'MOM'], help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate. If set to a negative value a learning rate '
                                                            'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.005) #0.05

    parser.add_argument('--learning_rate_decay_epochs', type=int,  help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float, help='Learning rate decay factor.', default=1.0)   #1.0
    parser.add_argument('--moving_average_decay', type=float,    help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,   help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,  help='Number of preprocessing (data loading and augmentation) threads.', default=4) #4  process
    parser.add_argument('--log_histograms',     help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,   help='File containing the learning rate schedule that is used when '
                                                                          'learning_rate is set to to -1.', default='learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str, help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')    #not used
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])   #jpg
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')          #no input
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)  #3333333333333333333333333333333333333333333333333333333333333
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10) #33333333333333333333333333333333333333333333333333333
    return parser.parse_args(argv)
  
if __name__ == '__main__':             #if run current module, then execute __main__:
    main(parse_arguments(sys.argv[1:]))  #use  parse_arguments as default