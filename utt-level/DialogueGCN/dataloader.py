import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import builtins
import traceback
from sklearn.preprocessing import StandardScaler
class PulseEmoDataset(Dataset):
    
    def __init__(self, path, split='train'):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.sentences = data["Sentences"]
        self.speakers = data["Speakers"]
        self.subject_labels = data["SubjectLabels"]
        self.object_labels = data["ObjectLabels"]
        self.dialog_ids = data["Dialog"]
        self.textual = np.array(data["Textual"])    # (n_samples, textual_dim 768)
        self.acoustic = np.array(data["Acoustic"])  # (n_samples, acoustic_dim 65)
        self.visual = np.array(data["Visual"])      # (n_samples, visual_dim 713)
        self.physio =  np.array(data["Physio"]) 
        self.vid_split = data["Vid"]
        textual_scaler = StandardScaler()
        acoustic_scaler = StandardScaler()
        visual_scaler = StandardScaler()
        physio_scaler = StandardScaler()


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
            torch.FloatTensor([1]*len(self.subject_labels[vid])),\
            torch.LongTensor(self.subject_labels[vid]),\
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i], True) if i<7 else dat[i].tolist() for i in dat]



def group_by_dialog(data_list, dialog_ids):
    grouped = defaultdict(list)
    for item, dialog_id in zip(data_list, dialog_ids):
        grouped[dialog_id].append(item)
    return dict(grouped)      
from collections import defaultdict       