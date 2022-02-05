"""
Created on Sun Jan 21 2022

main

@author: Ray
"""
from networks import network
from config import params
from utils import *
from argparse import ArgumentParser, RawTextHelpFormatter
import sys


parser = ArgumentParser(usage=None, formatter_class=RawTextHelpFormatter,
    description="Super resolution implementation using generative adverserial network: \n \n"
    "This code provides a noval GAN structure for super resolution tasks. "
    "Users can change configuration via config.py. \n")

parser.add_argument("mode", help="train or test")
parser.add_argument("-f", "--filename", type=str, default=None, dest="filename")
args = parser.parse_args()

def main():

    image_path = args.filename if os.path.isfile(args.filename) else sys.exit("Incorrect file")
    if args.mode.lower() == 'train':

        with open(image_path) as f: 
            image_lists = f.read().splitlines()

        for i in image_lists:
            g_train, d_train = i.split(', ')
            sys.exit("Found wrong path or wrong format") if os.path.isfile(d_train) == False else None
            sys.exit("Found wrong path or wrong format") if os.path.isfile(g_train) == False else None

        print("All images are loaded")
                
        with tf.Session() as sess:
            model = network(sess, params)
            model.train(image_lists)
            
    elif args.mode.lower() == 'test':
        folder = input("Please input dir containing images: ")
        paths = load_path(folder)
        sys.exit("There is no image in this dir") if paths == [] else None

        with tf.Session() as sess:
            model = network(sess, params)
            g_imgs = model.test(paths)
        output_path = save_data(g_imgs, paths, params['save_path'])
        print('Saved in {}'.format(params['save_path']))

    else:
        sys.exit("Incorrect mode")
    
if __name__ == '__main__':
    main()