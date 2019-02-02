''' Frustum PointNets v2 Model.
'''
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss


def get_instance_seg_v2_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points, name_prefix=''):
    ''' 3D instance segmentation PointNet v2 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        end_points: dict
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
        end_points: dict
    '''

    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,1])

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points,
        128, [0.2,0.4,0.8], [32,64,128],
        [[32,32,64], [64,64,128], [64,96,128]],
        is_training, bn_decay, scope=name_prefix + 'layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points,
        32, [0.4,0.8,1.6], [64,64,128],
        [[64,64,128], [128,128,256], [128,128,256]],
        is_training, bn_decay, scope=name_prefix + 'layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[128,256,1024],
        mlp2=None, group_all=True, is_training=is_training,
        bn_decay=bn_decay, scope=name_prefix + 'layer3')

    # Feature Propagation layers
    l3_points = tf.concat([l3_points, tf.expand_dims(one_hot_vec, 1)], axis=2)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
        [128,128], is_training, bn_decay, scope=name_prefix + 'fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
        [128,128], is_training, bn_decay, scope=name_prefix + 'fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
        tf.concat([l0_xyz,l0_points],axis=-1), l1_points,
        [128,128], is_training, bn_decay, scope=name_prefix + 'fa_layer3')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
        is_training=is_training, scope=name_prefix + 'conv1d-fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.7,
        is_training=is_training, scope=name_prefix + 'dp1')
    logits = tf_util.conv1d(net, 2, 1,
        padding='VALID', activation_fn=None, scope=name_prefix + 'conv1d-fc2')

    return logits, end_points

def get_3d_box_estimation_v2_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points, name_prefix=''):
    ''' 3D Box Estimation PointNet v2 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            masked point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    ''' 
    # Gather object points
    batch_size = object_point_cloud.get_shape()[0].value

    l0_xyz = object_point_cloud
    l0_points = None
    # Set abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
        npoint=128, radius=0.2, nsample=64, mlp=[64,64,128],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope=name_prefix + 'ssg-layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
        npoint=32, radius=0.4, nsample=64, mlp=[128,128,256],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope=name_prefix + 'ssg-layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[256,256,512],
        mlp2=None, group_all=True,
        is_training=is_training, bn_decay=bn_decay, scope=name_prefix + 'ssg-layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, bn=True,
        is_training=is_training, scope=name_prefix + 'fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True,
        is_training=is_training, scope=name_prefix + 'fc2', bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope=name_prefix + 'fc3')
    return output, end_points


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None, config='EEE'):
    ''' Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.

    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
    Output:
        end_points: dict (map from name strings to TF tensors)
    '''
    end_points = {}
    
    # 3D Instance Segmentation PointNet
    logits_1, end_points = get_instance_seg_v2_net(
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points, 'branch1_')
    logits_2, end_points = get_instance_seg_v2_net(
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points, 'branch2_')
    logits_3, end_points = get_instance_seg_v2_net(
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points, 'branch3_')

    if config[0] == '1':
        logits = logits_1
    elif config[0] == '2':
        logits = logits_2
    elif config[0] == '3':
        logits = logits_3
    elif config[0] == 'E':
        logits = tf.reduce_mean(tf.stack((logits_1, logits_2, logits_3), axis=0), axis=0)

    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = \
        point_cloud_masking(point_cloud, logits, end_points)

    # T-Net and coordinate translation
    center_delta_1, end_points = get_center_regression_net(
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points, 'branch1_')
    center_delta_2, end_points = get_center_regression_net(
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points, 'branch2_')
    center_delta_3, end_points = get_center_regression_net(
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points, 'branch3_')

    if config[1] == '1':
        center_delta = center_delta_1
    elif config[1] == '2':
        center_delta = center_delta_2
    elif config[1] == '3':
        center_delta = center_delta_3
    elif config[1] == 'E':
        center_delta = tf.reduce_mean(tf.stack((center_delta_1, center_delta_2, center_delta_3), axis=0), axis=0)

    stage1_center = center_delta + mask_xyz_mean # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = \
        object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output_1, end_points = get_3d_box_estimation_v2_net(
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points, 'branch1_')
    output_2, end_points = get_3d_box_estimation_v2_net(
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points, 'branch2_')
    output_3, end_points = get_3d_box_estimation_v2_net(
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points, 'branch3_')

    if config[2] == '1':
        output = output_1
    elif config[2] == '2':
        output = output_2
    elif config[2] == '3':
        output = output_3
    elif config[2] == 'E':
        output = tf.reduce_mean(tf.stack((output_1, output_2, output_3), axis=0), axis=0)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

    return end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,4))
        outputs = get_model(inputs, tf.ones((32,3)), tf.constant(True))
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.zeros((32,1024),dtype=tf.int32),
            tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,3)), outputs)
        print(loss)
