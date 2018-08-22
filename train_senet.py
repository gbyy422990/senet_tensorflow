#coding:utf-8
#Bin GAO

import os
import tensorflow as tf
import SEResNet
from SEResNet import inference,make_train_op,loss_CE
import yaml
import glob
import numpy as np


def data_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, 0.7)
    image = tf.image.random_hue(image, 0.3)
    #设置随机的对比度
    tf.image.random_contrast(image,lower=0.3,upper=1.0)

    return image


def get_image(image_buffer, label_buffer, augmentation = True):
    image = tf.image.decode_jpeg(image_buffer,channels=3)
    image = tf.image.resize_images(image, (cfg['height'],cfg['width']))
    image.set_shape([cfg['height'], cfg['width'], cfg['channels']])

    label = tf.one_hot(label_buffer,cfg['num_classes'])

    if augmentation:
        image = data_augmentation(image)

    return image,label



def create_image_lists():
    cate = [cfg['trainingset_path'] + x for x in os.listdir(cfg['trainingset_path']) if os.path.isdir(cfg['trainingset_path'] + x)]
    data = []

    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('Reading the image %s' % (im))
            data.append({
                'filename':im,
                'label_name':os.path.basename(folder),
                'label_index':idx
            })


    ratio = cfg['ratio']
    s = np.int(len(data) * ratio)
    train = data[:s]
    val = data[s:]

    return train,val



def main():
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, cfg['width'], cfg['height'], cfg['channels']], name="X")
    y = tf.placeholder(tf.float32, shape=[None, cfg['num_classes']], name="y")
    mode = tf.placeholder(tf.bool, name="mode")


    train, val = create_image_lists()

    train_filename = [d['filename'] for d in train]
    train_label_index = [d['label_index'] for d in train]

    val_filename = [d['filename'] for d in val]
    val_label_index = [d['label_index'] for d in val]


    train_filename_in, train_label_index_in = tf.train.slice_input_producer([train_filename,train_label_index],shuffle=True)

    val_filename_in, val_label_index_in = tf.train.slice_input_producer([val_filename, val_label_index], shuffle=True)


    train_image_file = tf.read_file(train_filename_in)
    train_image, train_label = get_image(train_image_file,train_label_index_in,augmentation=True)



    val_image_file = tf.read_file(val_filename_in)
    val_image, val_label = get_image(val_image_file,val_label_index_in,augmentation=True)

    X_batch, y_batch = tf.train.shuffle_batch([train_image,train_label],
                                                    batch_size=cfg['batch_size'],
                                                    capacity=cfg['batch_size']*5,
                                                    min_after_dequeue=cfg['batch_size']*2,
                                                    allow_smaller_final_batch=True)


    X_val, y_val = tf.train.batch([val_image,val_label],
                                        batch_size=cfg['batch_size'],
                                        capacity=cfg['batch_size']*2,
                                        allow_smaller_final_batch=True)

    X_batch_op_1 = tf.cast(X_batch,tf.float32)
    X_batch_op = tf.reshape(X_batch_op_1,shape=[-1,cfg['height'],cfg['width'],cfg['channels']])
    #y_batch_op = tf.reshape(y_batch,shape=[cfg['batch_size']])
    y_batch_op = y_batch


    C1, C2, C3, C4, C5,logits_output,prob = inference(input=X,is_training=mode)

    loss = loss_CE(y_pred=logits_output,y_true=y)
    #print(prob)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(loss,learning_rate=cfg['LR'])

    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    tf.summary.scalar("loss", loss)

    tf.summary.image("image", X)
    '''tf.summary.image("C1", C1)
    tf.summary.image("C2", C2)
    tf.summary.image("C3", C3)
    tf.summary.image("C4", C4)
    tf.summary.image("C5", C5)'''


    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_summary_writer = tf.summary.FileWriter(cfg['train_logdir'], sess.graph)
        test_summary_writer = tf.summary.FileWriter(cfg['train_logdir'])

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=10)

        if os.path.exists(cfg['model_save_dir']) and tf.train.checkpoint_exists(cfg['model_save_dir']):
            latest_check_point = tf.train.latest_checkpoint(cfg['model_save_dir'])
            saver.restore(sess, latest_check_point)
        else:
            os.mkdir(cfg['model_save_dir'])

        try:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(cfg['epochs']):
                print("------------------------------------------")
                print("epoch %d" % epoch)

                for step in range(0,len(train),cfg['batch_size']):
                    X_batch, y_batch = sess.run([X_batch_op, y_batch_op])

                    _, loss_value, step_summary = sess.run([train_op, loss, summary_op],
                                                           feed_dict={X: X_batch,y: y_batch,mode: True})

                    print("step:{:d}, loss:{:.4f}".format(step,loss_value))

                    if step % 10 == 0:
                        train_summary_writer.add_summary(step_summary, epoch * len(train) + step)

                for step in range(0, len(val), cfg['batch_size']):
                    X_batch_val, y_batch_val = sess.run([X_val, y_val])

                    _, loss_value, step_summary = sess.run([train_op, loss, summary_op],
                                                           feed_dict={X: X_batch_val, y: y_batch_val, mode: False})

                    print("step:{:d}, loss:{:.4f}".format(step, loss_value))

                    if step % 10 == 0:
                        train_summary_writer.add_summary(step_summary, epoch * len(train) + step)

                if epoch % 2 == 0:
                    saver.save(sess, "{:s}/model{:d}.ckpt".format(cfg['model_save_dir'], epoch))
                    print('save model %d' % epoch)
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(cfg['model_save_dir']))


if __name__ == '__main__':
    with open('cfg.yml') as file:
        cfg = yaml.load(file)

    print(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu'])
    main()
