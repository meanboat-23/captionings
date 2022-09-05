import os
import torch
import cv2
from typing import Tuple, Dict, List
import argparse
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
import glob
import numpy as np
import shutil
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def video_to_frames(root_path, out_dir, output_dim):
    video = root_path
    vid = cv2.VideoCapture(video) #Open video file
    #os.mkdir(out_dir) #make only one dir
    os.makedirs(out_dir, exist_ok=True) #if I want to make more than one dirs then, use this
    
    count = 1
    success, frame = vid.read() #read a video in each frame, if read it successfully success = True, if not success = False
    while(success):
        frame = cv2.resize(frame, output_dim) #resize the frame to output_dim
        cv2.imwrite(f'{out_dir}/{count:03}.jpg', frame) #store the frame or image
        success, frame = vid.read() #read next frame
        count += 1

def extract_feats(params, model, preprocess, output_dim): 
    #model.eval()
    features_dir = params['output_dir'] #select the dir where frame features will be stored
    if not os.path.isdir(features_dir): #if the dir is not exist then, make a new dir in the path
        os.mkdir(features_dir)
    
    video_list = glob.glob(os.path.join(params['video_path'], '*.avi')) #return the file name that meets the conditions presented by the user in the form of a list

    for video in tqdm(video_list): #to show the progress bar
        video_id = video.split("/")[-1].split(".")[0] #variable name video contains like '/home/minbae/Desktop/sequence/YouTubeClips/-4wsuPCjDBc_5_15.avi' so I need only file name, '-4wsuPCjDBc_5_15'
        vid_frame = params['model'] + '_' + video_id #select the path where video frames will be stored
        video_to_frames(video,vid_frame, output_dim)

        image_list = sorted(glob.glob(os.path.join(vid_frame, '*.jpg'))) #return the file name that meets the conditions presented by the user in the form of a list
        samples = np.round(range(0, len(image_list) - 1, params['n_frame'])) #sample every tenth frame 
        image_list = [image_list[int(sample)] for sample in samples] #choose the images that have same index in the array
        images = torch.zeros((len(image_list), 3, 224, 224)) #pad with zeros.
        
        for real in range(len(image_list)):
            img = preprocess(Image.open(image_list[real])) #process each frames into given condition
            images[real] = img

        with torch.no_grad():
            out_feat = model(images.to(device)) #extract features
        out = out_feat.cpu().detach().numpy() #.cpu().numpy(): if tensor is worked on gpu then, need to move into cpu and convert into numpy, detach(): contain informations of gradient then, need to erase to convert into numpy
        outfile = os.path.join(features_dir, video_id + '.npy') #select the path where the features will be stored
        
        np.save(outfile, out) #save features as npy format
        shutil.rmtree(vid_frame) #delete all files and dirs in the path 


if __name__ == '__main__':
    parser = argparse.ArgumentParser() #parse the arguments and adds value
    parser.add_argument("--output_dir", dest='output_dir', type=str, default='/home/minbae/Desktop/sequence/output', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame', type=int, default=10, help='how many frames to sampler per video')
    parser.add_argument("--video_path", dest='video_path', type=str, default='/home/minbae/Desktop/sequence/YouTubeClips', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='vgg', help='the CNN model which is used to extract_feats')
    args = parser.parse_args()
    params = vars(args) #return into dict format

    if params['model'] == 'vgg': #to make vgg16 model
        model = models.vgg16(pretrained=True).to(device)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1]) #need to modify the model
        output_dim = (224,224)
    else:
        print("it doesn't support %s" % (params['model']))

    preprocess = transforms.Compose([ #in the paper they want this transform process
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    extract_feats(params, model, preprocess, output_dim) 