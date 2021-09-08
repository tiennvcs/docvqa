"""
Usage:

python pytesseract.py \
    --input_dir ./sample/ \
    --output_dir ./output
    --verbose True

"""

import os
import cv2
import argparse
import pytesseract
import numpy as np
from tqdm import tqdm
import glob2 


def ocr_on_image(path, output_dir, verbose=False):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract information from the image
    extract_data = pytesseract.image_to_data(img_rgb, lang='viet', output_type=pytesseract.Output.DICT)

    word_num_indices = np.where(np.array(extract_data['level'])==5)[0]


    # Write detection result to file
    with open(os.path.join(output_dir, 'text', os.path.basename(path).split(".")[0]+'.txt'), 'w') as f:
        for i in word_num_indices:
            line = "{}, {}, {}, {}, {}, {}, {}".format(
                extract_data['text'][i],
                extract_data['left'][i], 
                extract_data['top'][i], 
                extract_data['left'][i]+extract_data['width'][i], 
                extract_data['top'][i]+extract_data['height'][i],
                extract_data['conf'][i]/100, 0
            )
            f.write(line + '\n')

    if verbose:
        # Draw case 1
        for i in word_num_indices:
            (x, y, w, h) = (extract_data['left'][i], 
                            extract_data['top'][i], 
                            extract_data['width'][i], 
                            extract_data['height'][i])
            img_rgb = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(img, extract_data['text'][i], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        save_img_file = os.path.join(output_dir, 'images', os.path.basename(path))
    
        cv2.imwrite(save_img_file, img_rgb)
    

def main(args):
    
    # Check invalid input or not
    if not os.path.exists(args['input_dir']):
        print("INVALID INPUT DIRECTORY !")
        exit(0)
    if not os.path.exists(args['output_dir']):
        print("Creating output directory: {}".format(args['output_dir']))
        try:
            os.mkdir(args['output_dir'])
        except:
            print("INVALID OUTPUT DIRECTORY !")
            exit(0)
        os.mkdir(os.path.join(args['output_dir'], 'images'))
        os.mkdir(os.path.join(args['output_dir'], 'text'))

    if not os.path.exists(os.path.join(args['output_dir'], 'images')):
        os.mkdir(os.path.join(args['output_dir'], 'images'))
    if not os.path.exists(os.path.join(args['output_dir'], 'text')):
        os.mkdir(os.path.join(args['output_dir'], 'text'))

    img_paths = glob2.glob(os.path.join(args['input_dir'], '*.png'))
    # Extract information from each image
    for i, img_path in enumerate(img_paths):
        print("{:6}/{:6} Extracting {}".format(
            str(i).zfill(6), str(len(img_paths)).zfill(6), os.path.basename(img_path)))
        ocr_on_image(path=img_path, output_dir=args['output_dir'], verbose=args['verbose'])
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract information from image using Tesseract OCR engine.')
    parser.add_argument('--input_dir', required=True,
        help='The input folder contains images')
    parser.add_argument('--output_dir', required=True,
        help='The output directory to store extracted information')
    parser.add_argument('--verbose', action='store_true',
        help='Store image to disk wheter or not.')

    args = vars(parser.parse_args())

    print(args)

    main(args)

