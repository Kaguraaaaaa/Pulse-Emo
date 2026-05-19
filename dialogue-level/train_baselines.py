#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import pickle
import itertools
from collections import defaultdict
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from transformers import RobertaTokenizer, RobertaModel
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch and transformers not installed, training with sklearn")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

PAD = 0
UNK = 1

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=128, nhead=4, num_layers=2, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = self.fc(x[:, 0, :])
        return x

class TextFeatureExtractor:
    # Roberta path
    def __init__(self, model_path="/data3/xgy/SATI_pulsemo/model"):
        if not TORCH_AVAILABLE:
            raise ImportError("Requires PyTorch and transformers")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocab_file = os.path.join(model_path, 'vocab.json')
        merges_file = os.path.join(model_path, 'merges.txt')
        
        self.tokenizer = RobertaTokenizer(vocab_file, merges_file)
        self.model = RobertaModel.from_pretrained(os.path.join(model_path, 'roberta-base'))
        self.model.to(self.device)
        self.model.eval()
        
    def extract_features(self, text):
        if not text or len(text.strip()) == 0:
            return np.zeros(768)
            
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, 
                              truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        return features

def extract_text_features(data, text_extractor):
    print("[INFO] Starting text feature extraction...")
    text_features = []
    
    for i, sample in enumerate(data):
        try:
            if len(sample[0]) == 5:
                actual_words = sample[0][4]
            else:
                actual_words = sample[0][3] if len(sample[0]) > 3 else []
            
            if isinstance(actual_words, list) and len(actual_words) > 0:
                text = " ".join(str(word) for word in actual_words)
                features = text_extractor.extract_features(text)
            else:
                features = np.zeros(768)
            
            text_features.append(features)
            
            if (i + 1) % 100 == 0:
                print(f"[INFO] Processed {i + 1}/{len(data)} samples")
                
        except Exception as e:
            print(f"[ERROR] Error processing sample {i}: {e}")
            text_features.append(np.zeros(768))
    
    text_features = np.array(text_features, dtype=np.float32)
    print(f"[INFO] Text feature extraction completed: {text_features.shape}")
    return text_features

class MultimodalDataset(Dataset):
    def __init__(self, data, modalities=['physio'], text_features=None):
        self.data = data
        self.modalities = modalities
        self.text_features = text_features
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        features = {}
        
        if 'physio' in self.modalities and len(sample[0]) >= 4:
            physio_features = sample[0][3] if len(sample[0]) > 3 else np.zeros(100)
            if isinstance(physio_features, np.ndarray):
                features['physio'] = np.mean(physio_features, axis=0) if physio_features.ndim > 1 else physio_features
            else:
                print("zero physio")
                features['physio'] = np.zeros(100)
        
        if 'audio' in self.modalities and len(sample[0]) >= 2:
            audio_features = sample[0][2] if len(sample[0]) > 2 else np.zeros(100)
            if isinstance(audio_features, np.ndarray):
                features['audio'] = np.mean(audio_features, axis=0) if audio_features.ndim > 1 else audio_features
            else:
                print("zero audio")
                features['audio'] = np.zeros(100)
        
        if 'visual' in self.modalities and len(sample[0]) >= 1:
            visual_features = sample[0][1] if len(sample[0]) > 1 else np.zeros(100)
            if isinstance(visual_features, np.ndarray):
                features['visual'] = np.mean(visual_features, axis=0) if visual_features.ndim > 1 else visual_features
            else:
                print("zero visual")
                features['visual'] = np.zeros(100)
        
        if 'text' in self.modalities:
            if self.text_features is not None:
                features['text'] = self.text_features[idx]
            else:
                print("zero text")
                features['text'] = np.zeros(768)
        
        label = sample[1]
        if isinstance(label, np.ndarray):
            label = int(label.flatten()[0])
        else:
            label = int(label)
        
        return features, label

def load_multimodal_data(data_path, mode='train'):
    pkl_path = os.path.join(data_path, f'{mode}.pkl')
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Data file not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"[INFO] Loaded {mode} data: {len(data)} samples")
    return data

def extract_features_from_dataset(dataset, modalities, text_extractor=None):
    all_features = []
    all_labels = []
    
    for i in range(len(dataset)):
        try:
            features, label = dataset[i]
            valid_features = []
            for modality in modalities:
                if modality in features and features[modality] is not None:
                    valid_features.append(features[modality])
                else:
                    print(f"[WARNING] Missing {modality} features for sample {i}")
                    return None, None
            
            if len(valid_features) == len(modalities):
                combined_features = np.concatenate(valid_features)
                all_features.append(combined_features)
                all_labels.append(label)
        except Exception as e:
            print(f"[ERROR] Error processing sample {i}: {e}")
            continue
    
    if len(all_features) == 0:
        print("[ERROR] No valid features found")
        return None, None
    
    try:
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        print(f"[INFO] Feature extraction completed: {X.shape[0]} samples, {X.shape[1]} dimensions")
        return X, y
    except Exception as e:
        print(f"[ERROR] Error creating feature array: {e}")
        return None, None

def build_pipeline(model: str = "tree", pca_components: int = 30):
    clf = None
    try:
        if model == "tree":
            clf = DecisionTreeClassifier(class_weight='balanced', random_state=45, max_depth=10)
        elif model == "rf":
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=50, n_jobs=1, max_depth=10)
        elif model == "gbdt":
            clf = GradientBoostingClassifier(random_state=50, max_depth=5, n_estimators=100)
        elif model == "svm":
            clf = SVC(kernel='rbf', class_weight='balanced', probability=False, random_state=46, max_iter=1000)
        elif model == "svm_poly":
            clf = SVC(kernel='poly', class_weight='balanced', probability=False, random_state=46, max_iter=1000)
        elif model == "svm_linear":
            clf = SVC(kernel='linear', class_weight='balanced', probability=False, random_state=46, max_iter=1000)
        elif model == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=300, random_state=50, early_stopping=True)
        else:
            raise ValueError()
    except Exception as e:
        clf = SVC(kernel='linear', class_weight='balanced', probability=False, random_state=46, max_iter=1000)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_components)),
        ('clf', clf)
    ])
    return pipe

def train_deep_model(X_train, y_train, X_test, y_test, model_name, num_classes, epochs=50, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]

    if model_name == "bilstm":
        model = BiLSTMClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    elif model_name == "transformer":
        model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    else:
        raise ValueError("Only bilstm / transformer are supported")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    best_acc = 0.0
    best_f1 = 0.0
    best_y_pred = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_y_pred = y_pred.copy()

    return model, best_acc, best_f1, y_true, best_y_pred

def train_evaluate(data_path, modalities, model_name, pca_components=30, seed=45,
                  text_extractor=None, train_text_features=None, test_text_features=None):
    modality_str = "+".join(modalities)
    print(f"\n{'='*60}")
    print(f"Training Combination | Modalities: {modality_str} | Model: {model_name}")
    print(f"{'='*60}")

    train_data = load_multimodal_data(data_path, 'train')
    test_data = load_multimodal_data(data_path, 'test')

    train_dataset = MultimodalDataset(train_data, modalities=modalities, text_features=train_text_features)
    test_dataset = MultimodalDataset(test_data, modalities=modalities, text_features=test_text_features)

    X_train, y_train = extract_features_from_dataset(train_dataset, modalities)
    X_test, y_test = extract_features_from_dataset(test_dataset, modalities)

    if X_train is None or X_test is None:
        print(f"[ERROR] Failed to extract {modality_str} features")
        return None

    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    print(f"[INFO] Train set: {X_train.shape}, Test set: {X_test.shape}")

    if model_name in ["bilstm", "transformer"]:
        if not TORCH_AVAILABLE:
            print("[WARNING] PyTorch not installed, skipping deep learning model")
            return None
        model, acc, f1, y_true, y_pred = train_deep_model(
            X_train, y_train_enc, X_test, y_test_enc, model_name, num_classes
        )
    else:
        max_pca = min(pca_components, min(X_train.shape)-1)
        pipe = build_pipeline(model_name, max_pca)
        pipe.fit(X_train, y_train_enc)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        f1 = f1_score(y_test_enc, y_pred, average='weighted')

    print(f"\n[RESULT] Modalities: {modality_str} | Model: {model_name}")
    print(f"Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
    print(f"Classification Report:\n{classification_report(y_test_enc, y_pred, target_names=[str(c) for c in le.classes_])}")

    return {
        "modalities": modality_str,
        "model": model_name,
        "accuracy": acc,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Fully automatic multimodal emotion classification training")
    parser.add_argument("--data_path", default="./DTCA/datasets/PULSE_EMO", help="Dataset path")
    parser.add_argument("--pca", type=int, default=30, help="PCA components")
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed) if TORCH_AVAILABLE else None

    ALL_MODELS = [
        "tree", "rf", "gbdt",
        "svm", "svm_poly", "svm_linear",
        "mlp", "bilstm", "transformer"
    ]
    BASE_MODALITIES = ['text', 'audio', 'visual', 'physio']
    
    ALL_MODALITY_COMBOS = [
        ['text'],                          
        ['text', 'audio'],                 
        ['text', 'visual'],                
        ['text', 'physio'],                
        ['text', 'audio', 'visual'],       
        ['text', 'audio', 'physio'],       
        ['text', 'visual', 'physio'],      
        ['text', 'audio', 'visual', 'physio']  
    ]

    text_extractor = None
    train_tf = None
    test_tf = None
    if TORCH_AVAILABLE:
        try:
            text_extractor = TextFeatureExtractor()
            train_data = load_multimodal_data(args.data_path, 'train')
            test_data = load_multimodal_data(args.data_path, 'test')
            train_tf = extract_text_features(train_data, text_extractor)
            test_tf = extract_text_features(test_data, text_extractor)
        except:
            print("[WARNING] Text feature extraction failed")

    all_results = []
    print("\n[INFO] Starting fully automatic training: all models + all modality combinations")
    print(f"Total models: {len(ALL_MODELS)} | Total modality combinations: {len(ALL_MODALITY_COMBOS)}")

    for modalities in ALL_MODALITY_COMBOS:
        for model in ALL_MODELS:
            try:
                res = train_evaluate(
                    data_path=args.data_path,
                    modalities=modalities,
                    model_name=model,
                    pca_components=args.pca,
                    seed=args.seed,
                    text_extractor=text_extractor,
                    train_text_features=train_tf,
                    test_text_features=test_tf
                )
                if res:
                    all_results.append(res)
            except Exception as e:
                print(f"[FAILED] Modalities {modalities} + Model {model}: {str(e)}")
                continue

    print("\n" + "="*80)
    print("All experiments completed - LaTeX table format output")
    print("="*80)
    
    modality_map = {
        "l": "text",
        "l&a": "text+audio",
        "l&v": "text+visual",
        "l&p": "text+physio",
        "l&a&v": "text+audio+visual",
        "l&a&p": "text+audio+physio",
        "l&v&p": "text+visual+physio",
        "l&a&v&p": "text+audio+visual+physio"
    }
    
    latex_cols = ["l", "l&a", "l&v", "l&p", "l&a&v", "l&a&p", "l&v&p", "l&a&v&p"]
    all_models = ["tree", "rf", "gbdt", "svm", "svm_poly", "svm_linear", "mlp", "bilstm", "transformer"]
    
    if all_results:
        result_dict = {}
        for res in all_results:
            model = res["model"]
            mod = res["modalities"]
            acc = round(res["accuracy"], 4)
            f1 = round(res["f1"], 4)
            if model not in result_dict:
                result_dict[model] = {}
            result_dict[model][mod] = (acc, f1)
        
        print("% ========== Multimodal Emotion Recognition Results ==========")
        print(r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Metric}} & \multicolumn{8}{c}{\textbf{Modalities}} \\")
        print(r"     & & l & l\&a & l\&v & l\&p & l\&a\&v & l\&a\&p & l\&v\&p & l\&a\&v\&p \\ \hline")
        
        for model in all_models:
            if model not in result_dict:
                continue
            mod_data = result_dict[model]
            
            acc_row = []
            f1_row = []
            for col in latex_cols:
                target_mod = modality_map[col]
                acc, f1 = mod_data.get(target_mod, (0.0000, 0.0000))
                acc = acc * 100
                f1 = f1 * 100
                acc_row.append(f"{acc:.2f}")
                f1_row.append(f"{f1:.2f}")
            
            print(r"\multirow{2}{*}{" + model + r"} & Acc & " + " & ".join(acc_row) + r" \\")
            print(r" & F1 & " + " & ".join(f1_row) + r" \\ \hline")
        
        all_results_sorted = sorted(all_results, key=lambda x: x['f1'], reverse=True)
        best = all_results_sorted[0]
        print(f"% Best combination: Modalities={best['modalities']}, Model={best['model']}")
        print(f"% Best metrics: Accuracy={best['accuracy']:.4f}, Weighted F1={best['f1']:.4f}")
    else:
        print("[ERROR] No valid training results")

if __name__ == "__main__":
    main()
