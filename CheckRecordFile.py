import os
import cv2
import numpy as np
import tensorflow as tf
print(tf.__version__)

class CheckRecordFile:
    def __init__(self, input_path, save_image_path, record_path, manner):
        self.__save_image_path = save_image_path
        self.__input_path = input_path
        self.__record_path = record_path

        self.__manner = manner

        self.read_tfrecord(self.__input_path, self.__save_image_path, self.__record_path, self.__manner)

    def read_tfrecord(self, input_path, save_image_path, record_path, manner):
        raw_dataset = tf.data.TFRecordDataset(record_path)

        feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),

            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)
        #
        parsed_dataset = raw_dataset.map(_parse_function)
        # print(parsed_dataset)
        # input()

        # for raw_record in raw_dataset:
        #     example = tf.train.Example()
        #     example.ParseFromString(raw_record.numpy())
        #     print(example)
        #     input()

        # for parsed_record in parsed_dataset.take(1):
        #     print(repr(parsed_record))

        counter = 0
        for parsed_record in parsed_dataset:
            # Get Filename
            filename = parsed_record['image/filename'].numpy().decode('utf-8')
            # print(filename)
            # Image width
            width = parsed_record['image/width'].numpy()
            # print(width)
            # Image height
            height = parsed_record['image/height'].numpy()
            # print(height)
            xmin = parsed_record['image/object/bbox/xmin'].values.numpy() * width
            ymin = parsed_record['image/object/bbox/ymin'].values.numpy() * height
            xmax = parsed_record['image/object/bbox/xmax'].values.numpy() * width
            ymax = parsed_record['image/object/bbox/ymax'].values.numpy() * height

            class_name = parsed_record['image/object/class/text'].values.numpy()
            class_label = parsed_record['image/object/class/label'].values.numpy()

            for i in range(len(class_name)):
                class_name[i] = class_name[i].decode('utf-8')

            print(class_name)
            if type(class_name) == np.ndarray:
                class_name_ = str(class_name[0])
                class_label_ = class_label[0]

            img_path = os.path.join(input_path, manner, class_name_, filename)
            img = cv2.imread(img_path)
            for i in range(len(xmin)):
                start_point = (xmin[i], ymin[i])
                # Ending coordinate, here (220, 220)
                # represents the bottom right corner of rectangle
                end_point = (xmax[i], ymax[i])
                # Blue color in BGR
                color = (255, 255, 0)
                thickness = 1
                img = cv2.rectangle(img, start_point, end_point, color, thickness)

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 500)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2

                cv2.putText(img, str(class_name_) + '&' + str(class_label),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            cv2.imwrite(save_image_path + '/' + str(counter) + class_name_ + '.jpg', img)
            counter += 1