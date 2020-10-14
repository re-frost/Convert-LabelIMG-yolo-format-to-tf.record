import os
import glob
import pandas as pd
import io

import tensorflow as tf
print(tf.__version__)
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

class GenerateTFrecord():
    def __init__(self, input_path, output_path, pbtxt, manner):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__label_path = pbtxt
        self.__manner = manner

        self.__label_map = label_map_util.load_labelmap(self.__label_path)
        self.__label_map_dict = label_map_util.get_label_map_dict(self.__label_map)

        self.__column_name = ['filename', 'width', 'height', 'class_txt',
                            'class_num', 'xmin', 'ymin', 'xmax', 'ymax']
        self.__file_list = []
        self.list_image_label_per_class(self.__file_list, self.__column_name,
                                        self.__output_path, self.__input_path, self.__manner)

    def list_image_label_per_class(self, file_list, column_name,
                                   output_path, input_path, manner):
        writer = tf.io.TFRecordWriter(output_path)



        path_training_images = os.path.join(input_path, manner)
        subfolders = sorted(os.listdir(path_training_images))

        for subdir in subfolders:
            path_singel_class = os.path.join(path_training_images, subdir)
            image_list = sorted(glob.glob(path_singel_class + '/*.jpg'))
            label_list = sorted(glob.glob(path_singel_class + '/Label' + '/*.txt'))

            if len(image_list) != len(label_list):
                print("len(image_list) != len(label_list)")
                input()

            for i in range(len(image_list)):
                image_name = os.path.splitext(os.path.basename(image_list[i]))[0]
                label_name = os.path.splitext(os.path.basename(label_list[i]))[0]

                if image_name != label_name:
                    print("image_name != label_name")
                    input()
                # open Image
                img = Image.open(image_list[i], 'r')

                current_txt = open(label_list[i], 'r').read().splitlines()
                for line in current_txt:
                    line = line.split(' ')

                    x_center = float(line[1]) * img.size[0]
                    y_center = float(line[2]) * img.size[1]
                    w_bb = float(line[3]) * img.size[0]
                    h_bb = float(line[4]) * img.size[1]

                    x_min = x_center - (w_bb / 2)
                    y_min = y_center - (h_bb / 2)
                    x_max = x_center + (w_bb / 2)
                    y_max = y_center + (h_bb / 2)

                    value = (str(os.path.basename(image_list[i])),
                             int(img.size[0]),
                             int(img.size[1]),
                             str(subdir),
                             int(line[0]) + 1,
                             float(x_min),
                             float(y_min),
                             float(x_max),
                             float(y_max)
                             )
                    file_list.append(value)

            file_list_df = pd.DataFrame(file_list, columns=column_name)
            file_list.clear()
            grouped = self.split(file_list_df, 'filename')
            for group in grouped:
                tf_example = self.create_tf_example(group, path_singel_class)
                writer.write(tf_example.SerializeToString())
            del file_list_df
        writer.close()

    def create_tf_example(self, group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class_txt'].encode('utf8'))
            classes.append(row['class_num'])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    # def class_text_to_int(self, row_label):
    #     return label_map_dict[row_label]

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]