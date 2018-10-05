import tensorflow as tf
import config
from crnn import CRNN
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Define if we train the model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we test the model"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="preprocess images"
    )
    return parser.parse_args()


def preprocess_img():
    utils.preprocess_imgs()

def build_model():
    crnn = CRNN(
            batch_size=config.BATCH_SIZE, model_path = config.MODEL_PATH,
            max_image_width = 150, restore = True, debug = True, phase = 'train'
        )
    return crnn

def main():
    args = parse_arguments()
    
    if args.preprocess:
        utils.preprocess_imgs()
    
    if args.train:
        crnn = build_model()
        crnn.train()
        
    if args.test:
        crnn = build_model()
        crnn.test()
    
    
if __name__ == '__main__':
    main()