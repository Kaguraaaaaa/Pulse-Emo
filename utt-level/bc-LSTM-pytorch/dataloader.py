import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'happy':0, 'sad':1, 'neutral':2, 'angry':3, 'excioted':4, 'frustrated':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence,\
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'),encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]

class MELDDataset(Dataset):

    def __init__(self, path, classify, train=True):
        self.videoIDs, self.videoSpeakers, self.emotion_labels, self.videoText,\
        self.videoAudio, self.videoSentence, self.trainVid,\
        self.testVid, self.sentiment_labels = pickle.load(open(path, 'rb'))
        
        if classify == 'emotion':
            self.videoLabels = self.emotion_labels
        else:
            self.videoLabels = self.sentiment_labels
        '''
        emotion_label_mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        setiment_label_mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]
        

class DailyDialogueDataset(Dataset):

    def __init__(self, path, split):
        
        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return torch.LongTensor(self.InputSequence[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.ActLabels[conv])), \
                torch.LongTensor(self.ActLabels[conv]), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                self.InputMaxSequenceLength[conv], \
                conv

    def __len__(self):
        return self.len

class DailyDialoguePadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # stack all
        return torch.stack(batch, dim=0)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i==0 else \
                pad_sequence(dat[i]) if i == 1 else \
                pad_sequence(dat[i], True) if i < 5 else \
                dat[i].tolist() for i in dat]
        
        
import builtins
import traceback
from sklearn.preprocessing import StandardScaler
_builtin_print = print
class PulseEmoDataset(Dataset):
    
    def __init__(self, path, split='train'):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.sentences = data["Sentences"]
        self.speakers = data["Speakers"]
        self.subject_labels = data["SubjectLabels"]
        self.object_labels = data["ObjectLabels"]
        self.dialog_ids = data["Dialog"]
        self.textual = np.array(data["Textual"])    # 形状: (n_samples, textual_dim 768)
        self.acoustic = np.array(data["Acoustic"])  # 形状: (n_samples, acoustic_dim 65)
        self.visual = np.array(data["Visual"])      # 形状: (n_samples, visual_dim 713)
        self.physio =  np.array(data["Physio"]) 
        self.vid_split = data["Vid"]
        textual_scaler = StandardScaler()
        acoustic_scaler = StandardScaler()
        visual_scaler = StandardScaler()
        physio_scaler = StandardScaler()

        # 拟合 + 变换
        self.textual = textual_scaler.fit_transform(data["Textual"])
        self.acoustic = acoustic_scaler.fit_transform(data["Acoustic"])
        self.visual = visual_scaler.fit_transform(data["Visual"])
        self.physio = physio_scaler.fit_transform(data["Physio"])
        """
        mapping labels
        """
        label_map = {
            'happy': 0,
            'sad': 1,
            'neutral': 2,
            'anger': 3,
            'surprise': 4,
            'fear': 5,
            'disgust': 6
        }

        def map_labels_flat(label_list):
            return [label_map[label] for label in label_list]
        #_builtin_print(len(self.keys))
        self.subject_labels = map_labels_flat(data["SubjectLabels"])
        self.object_labels = map_labels_flat(data["ObjectLabels"])
        self.sentences = group_by_dialog(self.sentences, self.dialog_ids)
        self.speakers = group_by_dialog(self.speakers, self.dialog_ids)
        self.subject_labels = group_by_dialog(self.subject_labels, self.dialog_ids)
        self.object_labels = group_by_dialog(self.object_labels, self.dialog_ids)
        self.textual = group_by_dialog(self.textual, self.dialog_ids)
        self.acoustic = group_by_dialog(self.acoustic, self.dialog_ids)
        self.visual = group_by_dialog(self.visual, self.dialog_ids)
        self.physio = group_by_dialog(self.physio, self.dialog_ids)
        self.dialog_ids = set(self.dialog_ids)
        #print("self.videoIDs", self.videoIDs.shape, "self.videoSpeakers", self.videoSpeakers.shape,  "self.videoL", self.videoLabels.shape, "self.videoT",self.videoText.shape)
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        
        '''
        
        self.vid_split = {dialog_id: split for dialog_id, split in zip(data["Dialog"], data["Vid"])}
        self.keys = [did for did in self.dialog_ids if self.vid_split[did] == split]
        #print(len(self.keys))
        #_builtin_print(len(self.keys))
        self.len = len(self.keys)
    def __getitem__(self, index):
        vid = self.keys[index]
        #_builtin_print(np.array(self.textual[vid]))

        return torch.FloatTensor(np.array(self.textual[vid])),\
            torch.FloatTensor(np.array(self.visual[vid])),\
            torch.FloatTensor(np.array(self.acoustic[vid])),\
            torch.FloatTensor(np.array(self.physio[vid])),\
            torch.FloatTensor([[1,0] if x=='A' else [0,1] for x in\
                                self.speakers[vid]]),\
            torch.FloatTensor([1]*len(self.object_labels[vid])),\
            torch.LongTensor(self.object_labels[vid]),\
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]




def group_by_dialog(data_list, dialog_ids):
    grouped = defaultdict(list)
    for item, dialog_id in zip(data_list, dialog_ids):
        grouped[dialog_id].append(item)
    return dict(grouped)      
from collections import defaultdict       
