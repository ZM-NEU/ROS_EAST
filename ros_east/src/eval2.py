#!/usr/bin/env python
import cv2
import time
import math
import os
import rospy
import numpy as np
import tensorflow as tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import locality_aware_nms as nms_locality
import lanms
import roslib;roslib.load_manifest('ros_east')  
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import Point32
from ros_east.msg import PolyImage
from std_msgs.msg import String
from std_msgs.msg import Header
tf.app.flags.DEFINE_string('test_data_path', '/home/zhouming/data/images_reach_720', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/zhouming/transition/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/home/zhouming/data/tmp/uml_01', '')
tf.app.flags.DEFINE_bool('no_write_images', True, 'do not write images')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS
global count



def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def callback(data):
    
    rospy.loginfo(rospy.get_caller_id() + "Image found %s", data.header.stamp)
    im = bridge.imgmsg_to_cv2(data, "bgr8")[:, :, ::-1]
    
    start_time = time.time()
    im_resized, (ratio_h, ratio_w) = resize_image(im)

    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()
    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
    timer['net'] = time.time() - start

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
  #  print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
#	im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

    if boxes is not None:
	boxes = boxes[:, :8].reshape((-1, 4, 2))
	boxes[:, :, 0] /= ratio_w
	boxes[:, :, 1] /= ratio_h
    duration = time.time() - start_time
    print('[timing] {}'.format(duration))

    # output detected text regions
    if boxes is not None:
	    #pols = Polys()

	    #
#	    img_msg = data
#	    img_msg.header.stamp = pols.header.stamp
#	    imgpub.publish(img_msg)
	    for box in boxes:
		# to avoid submitting errors
		box = sort_poly(box.astype(np.int32))
		if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
		    continue
	#	rospy.loginfo("Textbox xs found %s, %s, %s, %s",box[0, 0],box[1,0],box[2,0],box[3,0])
		pol_im = PolyImage()
		pol_im.header.stamp = rospy.Time.now()
		pt1 = Point32()
		pt2 = Point32()
		pt3 = Point32()
		pt4 = Point32()
		pl = Polygon()
		pt1.x = box[0,0]
		pt1.y = box[0,1]
		pl.points.append(pt1)
		pt2.x = box[1,0]
		pt2.y = box[1,1]
		pl.points.append(pt2)
		pt3.x = box[2,0]
		pt3.y = box[2,1]
		pl.points.append(pt3)
		pt4.x = box[3,0]
		pt4.y = box[3,1]
		pl.points.append(pt4)
		#pols.polygons.append(pl)
		pol_im.polygon = pl
		#publish images cut from detected text boxes
		img_cut = im[int(box[0,1]):int(box[2,1]),int(box[0,0]):int(box[2,0]),::-1]
		img_msg = bridge.cv2_to_imgmsg(img_cut,encoding="bgr8")
		img_msg.header.stamp = pol_im.header.stamp
		#imgpub.publish(img_msg)
		pol_im.imgpatch = img_msg
		pub.publish(pol_im)
		cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
		if not FLAGS.no_write_images:
		    sub_path = str(rospy.Time.now()) + ".png"
		    img_path = os.path.join(FLAGS.output_dir, sub_path)
		    rospy.loginfo("writing to: %s",img_path)
		    img_cut = im[int(box[0,1]):int(box[2,1]),int(box[0,0]):int(box[2,0])]
		    cv2.imwrite(img_path, img_cut[:, :, ::-1])
	    cv2.imshow("frame" , im)
	    cv2.waitKey(3)
	    #pub.publish(pols)
    else:
	#img_msg = data
	#img_msg.header.stamp = rospy.Time.now()
	#imgpub.publish(img_msg)
      cv2.imshow("frame" , im)
      cv2.waitKey(3)
  #  if not FLAGS.no_write_images:
#	img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
#	cv2.imwrite(img_path, im[:, :, ::-1])

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    global bridge,f_score, f_geometry,input_images,sess
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

       # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
	model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
	print('Restore from {}'.format(model_path))
	saver.restore(sess, model_path)   
	
    rospy.init_node('ros_east', anonymous=True)
 
    # make a video_object and init the video object
    bridge = CvBridge()
    rospy.Subscriber('/camera/infra1/image_rect_raw', Image, callback)
    global pub
    pub = rospy.Publisher('pol_im', PolyImage, queue_size=10)
    global imgpub
    #imgpub = rospy.Publisher('img_cut',Image, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    tf.app.run()
