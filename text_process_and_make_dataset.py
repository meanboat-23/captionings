import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_dictionary(annotation_file: str): # fucntion that make dictionary, video_list and bagofcaptions
    video_captions = {}     # list that store captions
    video_list = {}     # list that store video_list
    word2index = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2}     # vocab dictionary

    captions = open(annotation_file, 'r')       # open txt file
    line = captions.readline()      # read captions each sentence
    #total = 0
    while line:
        line = line.strip('\n')     # get rid of the new line symbol
        tokens = line.split(' ')        # split the setence into word by whitespace
        #total = total + len(tokens) - 1
        video_name = tokens[0]      # video_name is in the first index of the sentence
        vid_cap = [x.lower() for x in tokens[1:]]       # lower all the word
        vid_cap = ['<BOS>'] + vid_cap + ['<EOS>']       # add BOS, EOS token in the caption

        for word in vid_cap:        # put each word into dictionary
            if word not in word2index:      # if the word is not in the dictionary then, add it
                word2index[word] = len(word2index)

        if video_name is not video_list:        # put video_name in the list
            video_list[video_name] = len(video_list)

        if video_name in video_captions:        # make a bag of captions
            video_captions[video_name].append(vid_cap)
        else:
            video_captions[video_name] = [vid_cap]

        line = captions.readline()      # change into next line
    index2word = {index: word for word, index in word2index.items()}
    re_video_list = {index: name for name, index in video_list.items()}
    
    return (index2word, word2index, video_captions, re_video_list)

def indexfromsentence(sentence, word2index):        # return index, input is the word
    return [word2index[i] if i in word2index else -1 for i in sentence]

def sentencefromindex(indexes, index2word):     # return word, input is the index
    return [index2word[i] if i in index2word else -1 for i in indexes]

class preprocessdata(Dataset):
    def __init__(self, data_path, annotation_path):
        self.video_path = data_path
        self.videos = os.listdir(data_path)
        self.index2word, self.word2index, self.bagofcaptions, self.videowithindex = make_dictionary(annotation_path)
        self.vocabsize = len(self.index2word)
        
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_name = self.videowithindex[index+1]
        npy_path = os.path.join(self.video_path, video_name + '.npy')
        video_features = np.load(npy_path)
        features_data = torch.FloatTensor(video_features).to(device)

        sentence_index = random.randint(0, len(self.bagofcaptions[video_name]) - 1)
        sentencewithtoken = self.bagofcaptions[video_name][sentence_index]
        sentencewithpadding = sentencewithtoken + ['<EOS>'] * (80 - len(sentencewithtoken))

        annotation = indexfromsentence(sentencewithpadding, self.word2index)
        annotationTensor = torch.LongTensor(annotation).to(device)

        #one_hot_result = F.one_hot(annotationTensor, self.vocabsize).to(device)

        #return (features_data, one_hot_result)
        return (features_data, annotationTensor)