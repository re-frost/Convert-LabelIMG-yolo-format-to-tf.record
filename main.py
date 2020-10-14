import argparse
from GenerateTFrecord import GenerateTFrecord
from CheckRecordFile import CheckRecordFile

parser = argparse.ArgumentParser(
    description="Description follows")

parser.add_argument("-i",
                    "--INPUT_PATH",
                    help="Path to the folder where the data set is located. ",
                    type=str, default=None)
parser.add_argument("-o",
                    "--OUTPUT_PATH_TF_RECORD",
                    help="Path to the '.record' file should be stored. ",
                    type=str, default=None)
parser.add_argument("-pbtxt",
                    "--PBTXT",
                    help="Path to the label-map file. Should be a *.pbtxt file",
                    type=str, default=None)
parser.add_argument("-test",
                    "--TEST",
                    help="Create .record - file for test set.",
                    type=bool, default=False)
parser.add_argument("-check",
                    "--CHECK",
                    help="You will see",
                    type=bool, default=False)
parser.add_argument("-check_path",
                    "--CHECK_PATH",
                    help="Path to folder where the images will be saved.",
                    type=str, default=False)
args = parser.parse_args()



if __name__ == '__main__':
    print(args.INPUT_PATH)
    print(args.OUTPUT_PATH_TF_RECORD)
    print(args.PBTXT)
    print(args.TEST)
    print(args.CHECK)
    print(args.CHECK_PATH)

    if args.TEST == True:
        manner = 'test'
    else:
        manner = 'train'
    GenerateTFrecord(args.INPUT_PATH, args.OUTPUT_PATH_TF_RECORD, args.PBTXT, manner)
    print('Done')

    if args.CHECK == True:
        CheckRecordFile(args.INPUT_PATH, args.CHECK_PATH, args.OUTPUT_PATH_TF_RECORD, manner)
        # read_tfrecord("/home/felix/Pictures/test/test.record")

# -i /home/felix/Pictures/test \
# -o /home/felix/Pictures/test/test.record \
# -pbtxt /home/felix/Pictures/MobileNetv2_320x320/annotations/label_map.pbtxt \
# -test False \
# -check True \
# -check_path /home/felix/Pictures/test

