import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import pickle
import matplotlib.pyplot as plt



# 1、log信息共有四个等级，按重要性递增为：
# INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的）;
#
# 2、值的含义：不同值设置的是基础log信息（base_loging），运行时会输出base等级及其之上（更为严重）的信息。具体如下：
# 	base_loging 	屏蔽信息 	输出信息
# “0” 	INFO 	无 	INFO + WARNING + ERROR + FATAL
# “1” 	WARNING 	INFO 	WARNING + ERROR + FATAL
# “2” 	ERROR 	INFO + WARNING 	ERROR + FATAL
# “3” 	FATAL 	INFO + WARNING + ERROR 	FATAL
#
# 注意：
# 1、“0”为默认值，输出所有信息
# 2、设置为3时，不是说任何信息都不输出，ERROR之上还有FATAL
#

#Tensorflow Allocation Memory: Allocation of 38535168 exceeds 10% of system memory解决办法 下面三行
#内存溢出还可以降低卷积核的数目
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

''' 全局参数 '''
IMAGE_SIZE = 100
LEARNING_RATE =1e-4
TRAIN_STEP = 4000
TRAIN_SIZE = 100
TEST_STEP = 100
TEST_SIZE = 50
# batch的个数
BATCHS_NUM=40


#训练 True 测试False
IS_TRAIN =False

SAVE_PATH = './ralex_model/'

data_dir = './batch_files'
pic_path = './data/test'

''''''


def load_data(filename):
    '''从batch文件中读取图片信息'''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        return data['data'],data['label'],data['filenames']

# 读取数据的类
class InputData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        all_names = []
        for file in filenames:
            data, labels, filename = load_data(file)

            all_data.append(data)
            all_labels.append(labels)
            all_names += filename
        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_labels)


        self._filenames = all_names

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._indicator:
            self._shuffle_data()

    def _shuffle_data(self):
        # 把数据再混排
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        '''返回每一批次的数据'''
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('have no more examples')
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all examples')
        batch_data = self._data[self._indicator : end_indicator]
        batch_labels = self._labels[self._indicator : end_indicator]
        batch_filenames = self._filenames[self._indicator : end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels, batch_filenames

# 定义一个类
class MyTensor:
    def __init__(self):


        # 载入训练集和测试集
        train_filenames = [os.path.join(data_dir, 'train_batch_%d'%i) for i in range(1, BATCHS_NUM+1)]
        test_filenames = [os.path.join(data_dir, 'test_batch')]
        self.batch_train_data = InputData(train_filenames, True)
        self.batch_test_data = InputData(test_filenames, True)

        pass

    def flow(self):
        self.x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], 'input_data')
        self.y = tf.placeholder(tf.int64, [None], 'output_data')
        self.keep_prob = tf.placeholder(tf.float32)

        self.x = self.x / 255.0  #需不需要这一步？

        # 图片输入网络中
        fc = self.conv_net(self.x, self.keep_prob)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=fc)
        self.y_ = tf.nn.softmax(fc) # 计算每一类的概率
        self.predict = tf.argmax(fc, 1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.y), tf.float32))

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=1)

        print('计算流图已经搭建.')

    # 训练
    def myTrain(self):
        acc_list = []
        #---记录损失和train_acc--

        train_acc_plt=[]
        val_acc_plt=[]
        train_loss_plt=[]
        val_loss_plt=[]
        #-----------------------
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(TRAIN_STEP):
                train_data, train_label, _ = self.batch_train_data.next_batch(TRAIN_SIZE)

                eval_ops = [self.loss, self.acc, self.train_op]
                eval_ops_results = sess.run(eval_ops, feed_dict={
                    self.x:train_data,
                    self.y:train_label,
                    self.keep_prob:0.7
                })
                train_loss, train_acc = eval_ops_results[0:2]

                #-----开始记录----
                train_acc_plt.append(train_acc)
                train_loss_plt.append(train_loss)
                #-----------------
                acc_list.append(train_acc)


                test_acc_list = []

                test_data, test_label, _ = self.batch_test_data.next_batch(TRAIN_SIZE)
                val_loss,val_acc = sess.run([self.loss,self.acc],feed_dict={
                    self.x:test_data,
                    self.y:test_label,
                    self.keep_prob:1.0
                })
                test_acc_list.append(val_acc)

                # -----开始记录----
                val_acc_plt.append(val_acc)
                val_loss_plt.append(val_loss)
                # -----------------

                if (i + 1) % 100 == 0:
                    acc_mean = np.mean(acc_list)
                    print('step:{0},train_loss:{1:.5},train_acc:{2:.5},train_acc_mean:{3:.5}'.format(
                        i + 1, train_loss, train_acc, acc_mean
                    ))
                    print('[Test ] step:{0},val_loss:{1:.5},val_acc:{2:.5},val_acc_mean:{3:.5}'.format(
                        i+1,val_loss,val_acc,np.mean(test_acc_list)
                    ))
            # 保存训练后的模型
            os.makedirs(SAVE_PATH, exist_ok=True)
            self.saver.save(sess, SAVE_PATH + 'my_model.ckpt')

            #------------开始画图-----------

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以解释中文无法显示的问题
            font = {'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 14,
                    }

            fig, ax1 = plt.subplots()
            lns1 = ax1.plot(np.arange(len(train_loss_plt)), train_loss_plt, label="train_loss")
            lns1_ = ax1.plot(np.arange(len(val_loss_plt)), val_loss_plt, label="val_loss")
            plt.legend(loc='upper right', prop=font, frameon=False)
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('loss')
            plt.savefig("loss.png")

            fig2, ax2 = plt.subplots()
            lns2 = ax2.plot(np.arange(len(train_acc_plt)), train_acc_plt, label="train_acc")
            lns2_ = ax2.plot(np.arange(len(val_acc_plt)), val_acc_plt, label="val_acc")
            plt.legend(loc='upper right', prop=font, frameon=False)
            ax2.set_xlabel('iteration')
            ax2.set_ylabel('training acc')
            plt.savefig("acc.png")

            plt.show()



    def myTest(self):
        with tf.Session() as sess:
            model_file = tf.train.latest_checkpoint(SAVE_PATH)
            model = self.saver.restore(sess, save_path=model_file)
            test_acc_list = []
            predict_list = []
            for j in range(TEST_STEP):
                test_data, test_label, test_name = self.batch_test_data.next_batch(TEST_SIZE)
                for each_data, each_label, each_name in zip(test_data, test_label, test_name):
                    acc_val, y__, pre, test_img_data = sess.run(
                        [self.acc, self.y_, self.predict, self.x],
                        feed_dict={
                            self.x:each_data.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3),
                            self.y:each_label.reshape(1),
                            self.keep_prob:1.0
                        }
                    )
                    predict_list.append(pre[0])
                    test_acc_list.append(acc_val)

                    # 把测试结果显示出来
                    self.compare_test(test_img_data, each_label, pre[0], y__[0], each_name)
            print('[Test ] mean_acc:{0:.5}'.format(np.mean(test_acc_list)))

    def compare_test(self, input_image_arr, input_label, output, probability, img_name):
        classes = ['normal', 'unnormal']
        if input_label == output:
            result = '正确'
        else:
            result = '错误'
        print('测试【{0}】,输入的label:{1}, 预测得是{2}:{3}的概率:{4:.5}, 输入的图片名称:{5}'.format(
            result,input_label, output,classes[output], probability[output], img_name
        ))

    def basic_block(self,x,filter_num,training,stride=1):
        conv1 = tf.layers.conv2d(x, filter_num, (3, 3), (stride,stride),padding='same', activation=None)
        bn1 = tf.layers.batch_normalization(conv1, training=training)
        conv1 = tf.nn.relu(bn1)

        conv2 = tf.layers.conv2d(conv1, filter_num, (3, 3), (1, 1), padding='same', activation=None)
        bn2 = tf.layers.batch_normalization(conv2, training=training)


        if stride!=1:
            identity=tf.layers.conv2d(x, filter_num, (1, 1),  (stride,stride), activation=None)
            output = tf.add(bn2, identity)
            output = tf.nn.relu(output)

            return output
        else:
            identity=x
            output = tf.add(bn2, identity)
            output = tf.nn.relu(output)

            return output



    def build_resblock(self, x,filter_num,training, blocks, stride=1):

        conv=self.basic_block(x,filter_num,training,stride)

        # just down sample one time
        for pre in range(1, blocks):
            conv=self.basic_block(conv,filter_num,training,stride=1)
        return conv



    def conv_net(self, x, keep_prob):


        # -----------------改进1---------------------
        if keep_prob == 1.0:  # 在输入参数的时候，训练设为0.7 测试设为1.0全部激活
            training = False
        else:
            training = True
        # -----------------改进1---------------------
        with tf.variable_scope("conv1"):
            conv1 = tf.layers.conv2d(x, 16, (3, 3),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     name='conv1')
            conv1=tf.nn.relu(conv1)
            lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn1")
            pool1 = tf.layers.max_pooling2d(lrn1, (2, 2), (2, 2), name='pool1')

        with tf.variable_scope("conv2"):
            conv2 = tf.layers.conv2d(pool1, 32, (3, 3),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     name='conv2')
            conv2 = tf.nn.relu(conv2)
            lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn2")
            pool2 = tf.layers.max_pooling2d(lrn2, (2, 2), (2, 2), name='pool2')

        with tf.variable_scope("conv3"):
            conv3 = tf.layers.conv2d(pool2, 64, (3, 3),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     name='conv3')
            conv3 = tf.nn.relu(conv3)

        with tf.variable_scope("conv4"):
            conv4 = tf.layers.conv2d(conv3, 128, (3, 3),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     name='conv4')
            conv4 = tf.nn.relu(conv4)

        with tf.variable_scope("conv5"):
            conv5 = tf.layers.conv2d(conv4, 256, (3, 3),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     name='conv5')
            conv5 = tf.nn.relu(conv5)

        with tf.variable_scope("res1"):
            x_shortcut=conv5
            conv = tf.layers.conv2d(conv5, 256, (3, 3),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     name='res_conv1')
            bn1 = tf.layers.batch_normalization(conv, training=training)
            conv = tf.nn.relu(bn1)

            conv = tf.layers.conv2d(conv, 256, (3, 3),
                                    padding='same',
                                    activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='res_conv2')
            res_conv=tf.add(x_shortcut,conv)
            conv = tf.nn.relu(res_conv)

        with tf.variable_scope("res2"):
            x_shortcut = conv
            conv = tf.layers.conv2d(conv5, 256, (3, 3),
                                    padding='same',
                                    activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='res_conv3')
            bn1 = tf.layers.batch_normalization(conv, training=training)
            conv = tf.nn.relu(bn1)

            conv = tf.layers.conv2d(conv, 256, (3, 3),
                                    padding='same',
                                    activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='res_conv4')
            res_conv = tf.add(x_shortcut, conv)
            conv = tf.nn.relu(res_conv)

        with tf.variable_scope("res3"):
            x_shortcut = conv
            conv = tf.layers.conv2d(conv5, 256, (3, 3),
                                    padding='same',
                                    activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='res_conv5')
            bn1 = tf.layers.batch_normalization(conv, training=training)
            conv = tf.nn.relu(bn1)

            conv = tf.layers.conv2d(conv, 256, (3, 3),
                                    padding='same',
                                    activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='res_conv6')
            res_conv = tf.add(x_shortcut, conv)
            conv = tf.nn.relu(res_conv)

        pool = tf.layers.max_pooling2d(conv, (2, 2), (2, 2), name='pool5')

        flatten = tf.layers.flatten(pool)  # 把网络展平，以输入到后面的全连接层

        fc1 = tf.layers.dense(flatten, 2048, tf.nn.relu)
        if training == True:
            fc1_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)
            fc2 = tf.layers.dense(fc1_dropout, 1024, tf.nn.relu)
            fc2_dropout = tf.nn.dropout(fc2, keep_prob=keep_prob)
            fc3 = tf.layers.dense(fc2_dropout, 2, None)  # 得到输出fc3
        else:
            fc2 = tf.layers.dense(fc1, 1024, tf.nn.relu)
            fc3 = tf.layers.dense(fc2, 2, None)  # 得到输出fc3
        return fc3


    def main(self):
        self.flow()
        if IS_TRAIN is True:
            self.myTrain()
        else:
            self.myTest()

    def final_classify(self,out_test_path):#输入外部测试文件的路径
        #all_test_files_dir = './data/test'
        all_test_files_dir=out_test_path
        all_test_filenames = os.listdir(all_test_files_dir)
        if IS_TRAIN is False:
            self.flow()
            # self.classify()
            with tf.Session() as sess:
                model_file = tf.train.latest_checkpoint(SAVE_PATH)
                mpdel = self.saver.restore(sess,save_path=model_file)

                predict_list = []
                for each_filename in all_test_filenames:
                    each_data = self.get_img_data(os.path.join(all_test_files_dir,each_filename))
                    y__, pre, test_img_data = sess.run(
                        [self.y_, self.predict, self.x],
                        feed_dict={
                            self.x:each_data.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3),
                            self.keep_prob: 1.0
                        }
                    )
                    predict_list.append(pre[0])
                    self.classify(test_img_data, pre[0], y__[0], each_filename)

        else:
            print('now is training model...')

    def classify(self, input_image_arr, output, probability, img_name):

        classes = ['normal','unnormal']
        single_image = input_image_arr[0] #* 255
        if output == 0:
            output_dir = 'normal/'
        else:
            output_dir = 'unnormal/'
        os.makedirs(os.path.join('./classiedResult', output_dir), exist_ok=True)

        #删除文件夹下的所有文件
        self.del_files('./classiedResult')

        cv.imwrite(os.path.join('./classiedResult',output_dir, img_name),single_image)
        print('输入的图片名称:{0}，预测得有{1:5}的概率是{2}:{3}'.format(
            img_name,
            probability[output],
            output,
            classes[output]
        ))
    #删除文件夹下的所有文件
    def del_files(self,path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                self.del_files(f_path)
            else:
                os.remove(f_path)



    # 根据名称获取图片像素
    def get_img_data(self,img_name):
        img = cv.imread(img_name)
        resized_img = cv.resize(img, (100, 100))
        img_data = np.array(resized_img)

        return img_data




if __name__ == '__main__':

    mytensor = MyTensor()
    #mytensor.main()  # 用于训练或测试


    # 输入外部测试文件的路径
    out_test_path='./data/test/normal'
    mytensor.final_classify(out_test_path) # 用于最后的分类
