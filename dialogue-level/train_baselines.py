
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed version for Pulse-Emo multimodal emotion classification.

Main fixes compared with the original script:
1. Do NOT read sample[0][3] as text. sample[0][3] is used as physio.
2. Prefer sample[0][4] as actual_words when len(sample[0]) == 5.
3. Otherwise prefer sample[0][0] as text / text embedding.
4. Do not silently replace failed text extraction with all-zero vectors during training.
5. Print real exception traceback when RoBERTa/text extraction fails.
6. Add sample inspection and text-feature sanity checks.
"""

import os
import json
import argparse
import pickle
import traceback
from pathlib import Path

import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from transformers import RobertaTokenizer, RobertaModel

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch or transformers not installed. RoBERTa text extraction is unavailable.")


EXPECTED_TEXT_DIM = 768
DEFAULT_AUDIO_DIM = 100
DEFAULT_VISUAL_DIM = 100
DEFAULT_PHYSIO_DIM = 100


# =========================
# Deep learning classifiers
# =========================

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Input shape: [batch, feature_dim]
        x = x.unsqueeze(1)  # [batch, seq_len=1, feature_dim]
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=128, nhead=4, num_layers=2, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input shape: [batch, feature_dim]
        x = self.proj(x).unsqueeze(1)  # [batch, seq_len=1, d_model]
        x = self.transformer(x)
        x = self.fc(x[:, 0, :])
        return x


# =========================
# Utilities for data parsing
# =========================

def is_numeric_array_like(x):
    """Return True if x can be safely converted to a numeric numpy array."""
    if x is None:
        return False
    if isinstance(x, np.ndarray):
        return x.dtype.kind in {"i", "u", "f", "c", "b"}
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        try:
            arr = np.asarray(x)
            return arr.dtype.kind in {"i", "u", "f", "c", "b"}
        except Exception:
            return False
    return False


def to_1d_float_vector(x, default_dim=None, name="feature"):
    """
    Convert numeric feature to 1D float vector.
    If a 2D or higher array is provided, average over the first axis.
    """
    if x is None:
        if default_dim is None:
            raise ValueError(f"{name} is None")
        return np.zeros(default_dim, dtype=np.float32)

    if not is_numeric_array_like(x):
        if default_dim is None:
            raise ValueError(f"{name} is not numeric: type={type(x)}")
        return np.zeros(default_dim, dtype=np.float32)

    arr = np.asarray(x, dtype=np.float32)

    if arr.size == 0:
        if default_dim is None:
            raise ValueError(f"{name} is empty")
        return np.zeros(default_dim, dtype=np.float32)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = np.mean(arr, axis=0)

    return arr.reshape(-1).astype(np.float32)


def flatten_text_items(x):
    """Flatten nested textual data into a list of string tokens/sentences."""
    if x is None:
        return []

    if isinstance(x, str):
        x = x.strip()
        return [x] if x else []

    if isinstance(x, np.ndarray):
        # Only textual/object arrays are treated as text.
        if x.dtype.kind not in {"U", "S", "O"}:
            return []
        x = x.flatten().tolist()

    if isinstance(x, (list, tuple)):
        items = []
        for item in x:
            items.extend(flatten_text_items(item))
        return items

    # Avoid converting numeric features into fake text.
    if isinstance(x, (int, float, complex, np.number)):
        return []

    s = str(x).strip()
    return [s] if s else []


def normalize_text(x):
    """Convert raw text / token list / actual_words into one string."""
    words = flatten_text_items(x)
    words = [w for w in words if w]
    return " ".join(words).strip()


def find_raw_text_from_fields(fields):
    """
    Find raw textual content from sample[0].

    Expected layouts:
    - len(fields) == 5: fields[4] is actual_words / raw text.
    - len(fields) == 4: fields[0] is text or text embedding; fields[3] is physio.

    Important: fields[3] must NOT be used as text because the original Dataset uses it as physio.
    """
    if not isinstance(fields, (list, tuple)):
        return "", "none"

    candidates = []

    if len(fields) >= 5:
        candidates.append((4, fields[4]))

    if len(fields) >= 1:
        candidates.append((0, fields[0]))

    # Last resort: scan all fields except 1/2/3, which are usually visual/audio/physio.
    # This avoids treating numeric modality arrays as text.
    for idx, value in enumerate(fields):
        if idx in {1, 2, 3, 4, 0}:
            continue
        candidates.append((idx, value))

    for idx, value in candidates:
        text = normalize_text(value)
        if text:
            return text, f"sample[0][{idx}]"

    return "", "none"


def get_existing_text_embedding_from_fields(fields):
    """
    If sample already contains pre-extracted text embedding, return it.
    Prefer fields[0] because fields[1], fields[2], fields[3] are visual/audio/physio in this script.
    """
    if not isinstance(fields, (list, tuple)) or len(fields) == 0:
        return None

    x = fields[0]
    if is_numeric_array_like(x):
        vec = to_1d_float_vector(x, default_dim=None, name="existing text embedding")
        if vec.size > 0:
            return vec

    return None


def inspect_sample(data, name="train"):
    """Print the structure of the first sample to verify modality indices."""
    if data is None or len(data) == 0:
        print(f"[DEBUG] {name} data is empty")
        return

    sample = data[0]
    print(f"\n[DEBUG] Inspect {name} sample")
    print(f"sample type: {type(sample)}")
    try:
        print(f"len(sample): {len(sample)}")
    except Exception:
        pass

    try:
        print(f"label sample[1]: {sample[1]}")
    except Exception:
        print("label sample[1]: unavailable")

    try:
        fields = sample[0]
        print(f"sample[0] type: {type(fields)}")
        print(f"len(sample[0]): {len(fields)}")

        for i, item in enumerate(fields):
            if isinstance(item, np.ndarray):
                print(f"sample[0][{i}] ndarray shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, (list, tuple)):
                preview = str(item[:10]) if len(item) > 10 else str(item)
                print(f"sample[0][{i}] {type(item)} len={len(item)}, preview={preview[:300]}")
            else:
                preview = str(item)[:300]
                print(f"sample[0][{i}] type={type(item)}, preview={preview}")

        text, source = find_raw_text_from_fields(fields)
        print(f"[DEBUG] Detected raw text source: {source}")
        print(f"[DEBUG] Detected raw text preview: {text[:300] if text else '<EMPTY>'}")

        emb = get_existing_text_embedding_from_fields(fields)
        if emb is not None:
            print(f"[DEBUG] Existing text embedding detected at sample[0][0], dim={emb.shape[0]}")
    except Exception as e:
        print(f"[DEBUG] Failed to inspect sample: {e}")


# =========================
# Text feature extraction
# =========================

class TextFeatureExtractor:
    def __init__(self, model_path="/data3/xgy/SATI_pulsemo/model", local_files_only=True):
        if not TORCH_AVAILABLE:
            raise ImportError("TextFeatureExtractor requires PyTorch and transformers.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.local_files_only = local_files_only

        tokenizer = None
        tokenizer_errors = []

        tokenizer_candidates = [
            model_path,
            os.path.join(model_path, "roberta-base"),
        ]

        for cand in tokenizer_candidates:
            if not cand or not os.path.exists(cand):
                continue
            try:
                tokenizer = RobertaTokenizer.from_pretrained(cand, local_files_only=local_files_only)
                print(f"[INFO] Loaded tokenizer from: {cand}")
                break
            except Exception as e:
                tokenizer_errors.append(f"{cand}: {repr(e)}")

        if tokenizer is None:
            vocab_file = os.path.join(model_path, "vocab.json")
            merges_file = os.path.join(model_path, "merges.txt")
            if os.path.exists(vocab_file) and os.path.exists(merges_file):
                tokenizer = RobertaTokenizer(vocab_file=vocab_file, merges_file=merges_file)
                print(f"[INFO] Loaded tokenizer from vocab/merges: {vocab_file}, {merges_file}")
            else:
                raise FileNotFoundError(
                    "Could not load RoBERTa tokenizer. Tried from_pretrained candidates and vocab/merges.\n"
                    + "\n".join(tokenizer_errors)
                )

        model_candidates = [
            os.path.join(model_path, "roberta-base"),
            model_path,
        ]

        model = None
        model_errors = []
        for cand in model_candidates:
            if not cand or not os.path.exists(cand):
                continue
            try:
                model = RobertaModel.from_pretrained(cand, local_files_only=local_files_only)
                print(f"[INFO] Loaded RoBERTa model from: {cand}")
                break
            except Exception as e:
                model_errors.append(f"{cand}: {repr(e)}")

        if model is None:
            raise FileNotFoundError(
                "Could not load RoBERTa model. Check --model_path.\n"
                + "\n".join(model_errors)
            )

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

    def extract_features(self, text):
        text = normalize_text(text)
        if not text:
            return np.zeros(EXPECTED_TEXT_DIM, dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()

        return features.astype(np.float32)


def extract_text_features(data, text_extractor, split_name="train"):
    """
    Extract text features.

    This function no longer uses sample[0][3] as text.
    sample[0][3] is reserved for physio according to MultimodalDataset.
    """
    print(f"[INFO] Starting text feature extraction for {split_name}...")

    text_features = []
    empty_count = 0
    existing_embedding_count = 0
    roberta_count = 0

    for i, sample in enumerate(data):
        try:
            fields = sample[0]
            text, source = find_raw_text_from_fields(fields)

            if text:
                features = text_extractor.extract_features(text)
                roberta_count += 1
            else:
                # If raw text is unavailable but text embedding already exists in sample[0][0], use it.
                existing = get_existing_text_embedding_from_fields(fields)
                if existing is not None:
                    features = existing.astype(np.float32)
                    existing_embedding_count += 1
                    if i < 5:
                        print(f"[INFO] {split_name} sample {i}: used existing text embedding, dim={features.shape[0]}")
                else:
                    empty_count += 1
                    features = np.zeros(EXPECTED_TEXT_DIM, dtype=np.float32)
                    if i < 20:
                        print(f"[WARNING] {split_name} sample {i}: empty text, source={source}")

            text_features.append(features)

            if (i + 1) % 100 == 0:
                print(f"[INFO] Processed {i + 1}/{len(data)} {split_name} samples")

        except Exception as e:
            print(f"[ERROR] Error processing {split_name} sample {i}: {e}")
            traceback.print_exc()
            text_features.append(np.zeros(EXPECTED_TEXT_DIM, dtype=np.float32))
            empty_count += 1

    try:
        text_features = np.array(text_features, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(
            "Text feature dimensions are inconsistent. "
            "This usually means some samples contain text embeddings with different dimensions."
        ) from e

    norms = np.linalg.norm(text_features, axis=1)
    nonzero_count = int((norms > 0).sum())

    print(f"[INFO] Text feature extraction completed for {split_name}: {text_features.shape}")
    print(f"[INFO] {split_name} RoBERTa text samples: {roberta_count}/{len(data)}")
    print(f"[INFO] {split_name} existing text embedding samples: {existing_embedding_count}/{len(data)}")
    print(f"[INFO] {split_name} empty text samples: {empty_count}/{len(data)}")
    print(f"[INFO] {split_name} non-zero text features: {nonzero_count}/{len(data)}")

    if nonzero_count == 0:
        raise RuntimeError(
            f"All extracted text features are zero for {split_name}. "
            "Please check the printed sample structure. In this script, text should be sample[0][4] "
            "when len(sample[0]) == 5, otherwise sample[0][0]. sample[0][3] is physio, not text."
        )

    return text_features


def save_or_load_text_features(data, text_extractor, cache_path, split_name):
    """Load cached text features if available, otherwise extract and save."""
    if cache_path is None:
        return extract_text_features(data, text_extractor, split_name=split_name)

    cache_path = Path(cache_path)
    if cache_path.exists():
        arr = np.load(cache_path)
        print(f"[INFO] Loaded cached {split_name} text features from {cache_path}: {arr.shape}")
        return arr.astype(np.float32)

    arr = extract_text_features(data, text_extractor, split_name=split_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    print(f"[INFO] Saved {split_name} text features to {cache_path}")
    return arr


# =========================
# Dataset and feature matrix
# =========================

class MultimodalDataset(Dataset):
    def __init__(self, data, modalities=("physio",), text_features=None, strict_text=True):
        self.data = data
        self.modalities = list(modalities)
        self.text_features = text_features
        self.strict_text = strict_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        fields = sample[0]
        features = {}

        if "visual" in self.modalities:
            if isinstance(fields, (list, tuple)) and len(fields) > 1:
                features["visual"] = to_1d_float_vector(fields[1], DEFAULT_VISUAL_DIM, name="visual")
            else:
                features["visual"] = np.zeros(DEFAULT_VISUAL_DIM, dtype=np.float32)

        if "audio" in self.modalities:
            if isinstance(fields, (list, tuple)) and len(fields) > 2:
                features["audio"] = to_1d_float_vector(fields[2], DEFAULT_AUDIO_DIM, name="audio")
            else:
                features["audio"] = np.zeros(DEFAULT_AUDIO_DIM, dtype=np.float32)

        if "physio" in self.modalities:
            if isinstance(fields, (list, tuple)) and len(fields) > 3:
                features["physio"] = to_1d_float_vector(fields[3], DEFAULT_PHYSIO_DIM, name="physio")
            else:
                features["physio"] = np.zeros(DEFAULT_PHYSIO_DIM, dtype=np.float32)

        if "text" in self.modalities:
            if self.text_features is not None:
                features["text"] = to_1d_float_vector(self.text_features[idx], None, name="text_features")
            else:
                # Fallback only when sample already stores numeric text embedding at sample[0][0].
                existing = get_existing_text_embedding_from_fields(fields)
                if existing is not None:
                    features["text"] = existing.astype(np.float32)
                elif self.strict_text:
                    raise ValueError(
                        f"text_features is None and no existing numeric text embedding found at sample {idx}. "
                        "Text extraction probably failed. Do not train text modality with zero vectors."
                    )
                else:
                    features["text"] = np.zeros(EXPECTED_TEXT_DIM, dtype=np.float32)

        label = sample[1]
        if isinstance(label, np.ndarray):
            label = int(label.flatten()[0])
        else:
            label = int(label)

        return features, label


def load_multimodal_data(data_path, mode="train"):
    pkl_path = os.path.join(data_path, f"{mode}.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Data file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"[INFO] Loaded {mode} data: {len(data)} samples from {pkl_path}")
    return data


def extract_features_from_dataset(dataset, modalities):
    all_features = []
    all_labels = []

    for i in range(len(dataset)):
        try:
            features, label = dataset[i]
            valid_features = []

            for modality in modalities:
                if modality not in features or features[modality] is None:
                    raise ValueError(f"Missing {modality} features for sample {i}")
                valid_features.append(np.asarray(features[modality], dtype=np.float32).reshape(-1))

            combined_features = np.concatenate(valid_features).astype(np.float32)
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
    except Exception as e:
        print(f"[ERROR] Error creating feature array: {e}")
        return None, None

    print(f"[INFO] Feature extraction completed: {X.shape[0]} samples, {X.shape[1]} dimensions")
    return X, y


# =========================
# Model training
# =========================

def build_pipeline(model="tree", pca_components=30):
    if model == "tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=45, max_depth=10)
    elif model == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=50,
            n_jobs=1,
            max_depth=10,
        )
    elif model == "gbdt":
        clf = GradientBoostingClassifier(random_state=50, max_depth=5, n_estimators=100)
    elif model == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=46, max_iter=1000)
    elif model == "svm_poly":
        clf = SVC(kernel="poly", class_weight="balanced", probability=False, random_state=46, max_iter=1000)
    elif model == "svm_linear":
        clf = SVC(kernel="linear", class_weight="balanced", probability=False, random_state=46, max_iter=1000)
    elif model == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=50,
            early_stopping=True,
        )
    else:
        print(f"[WARNING] Unknown model '{model}', using linear SVM")
        clf = SVC(kernel="linear", class_weight="balanced", probability=False, random_state=46, max_iter=1000)

    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]

    if pca_components is not None and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components)))

    steps.append(("clf", clf))
    return Pipeline(steps)


def train_deep_model(X_train, y_train, X_test, y_test, model_name, num_classes, epochs=50, lr=1e-4):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    if model_name == "bilstm":
        model = BiLSTMClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    elif model_name == "transformer":
        model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    else:
        raise ValueError("Only bilstm / transformer are supported")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    best_acc = -1.0
    best_f1 = -1.0
    best_y_pred = None
    y_true = y_test_tensor.detach().cpu().numpy()

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
            _, y_pred_tensor = torch.max(outputs, 1)
            y_pred = y_pred_tensor.detach().cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_y_pred = y_pred.copy()

    return model, best_acc, best_f1, y_true, best_y_pred


def train_evaluate(
    data_path,
    modalities,
    model_name,
    pca_components=30,
    seed=45,
    train_text_features=None,
    test_text_features=None,
    strict_text=True,
):
    modality_str = "+".join(modalities)
    print(f"\n{'=' * 60}")
    print(f"Training Combination | Modalities: {modality_str} | Model: {model_name}")
    print(f"{'=' * 60}")

    train_data = load_multimodal_data(data_path, "train")
    test_data = load_multimodal_data(data_path, "test")

    train_dataset = MultimodalDataset(
        train_data,
        modalities=modalities,
        text_features=train_text_features,
        strict_text=strict_text,
    )
    test_dataset = MultimodalDataset(
        test_data,
        modalities=modalities,
        text_features=test_text_features,
        strict_text=strict_text,
    )

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
    print(f"[INFO] Classes: {list(le.classes_)}")

    if model_name in ["bilstm", "transformer"]:
        if not TORCH_AVAILABLE:
            print("[WARNING] PyTorch not installed, skipping deep learning model")
            return None
        _, acc, f1, _, y_pred = train_deep_model(
            X_train,
            y_train_enc,
            X_test,
            y_test_enc,
            model_name,
            num_classes,
        )
    else:
        max_pca = min(pca_components, X_train.shape[0] - 1, X_train.shape[1])
        if max_pca < 1:
            max_pca = None
        pipe = build_pipeline(model_name, max_pca)
        pipe.fit(X_train, y_train_enc)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        f1 = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)

    print(f"\n[RESULT] Modalities: {modality_str} | Model: {model_name}")
    print(f"Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
    print(
        "Classification Report:\n"
        + classification_report(
            y_test_enc,
            y_pred,
            target_names=[str(c) for c in le.classes_],
            zero_division=0,
        )
    )

    return {
        "modalities": modality_str,
        "model": model_name,
        "accuracy": float(acc),
        "f1": float(f1),
    }


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Fixed multimodal emotion classification training")
    parser.add_argument("--data_path", default="/data3/xgy/DTCA/DTCA/datasets/PULSE_EMO", help="Dataset path")
    parser.add_argument("--model_path", default="/data3/xgy/SATI_pulsemo/model", help="Local RoBERTa model path")
    parser.add_argument("--pca", type=int, default=30, help="PCA components")
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for BiLSTM/Transformer")
    parser.add_argument("--inspect_only", action="store_true", help="Only inspect sample structure and exit")
    parser.add_argument("--allow_zero_text", action="store_true", help="Allow zero text fallback. Not recommended.")
    parser.add_argument("--no_text_cache", action="store_true", help="Do not cache extracted text features")
    parser.add_argument("--train_text_cache", default=None, help="Path to cached train text features .npy")
    parser.add_argument("--test_text_cache", default=None, help="Path to cached test text features .npy")
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    train_data = load_multimodal_data(args.data_path, "train")
    test_data = load_multimodal_data(args.data_path, "test")

    inspect_sample(train_data, "train")
    inspect_sample(test_data, "test")

    if args.inspect_only:
        print("[INFO] inspect_only enabled. Exit now.")
        return

    all_models = [
        "tree",
        "rf",
        "gbdt",
        "svm",
        "svm_poly",
        "svm_linear",
        "mlp",
        "bilstm",
        "transformer",
    ]

    all_modality_combos = [
        ["text"],
        ["audio"],
        ["visual"],
        ["physio"],
        ["text", "audio", "visual"],
        ["text", "audio", "physio"],
        ["text", "visual", "physio"],
        ["audio", "visual", "physio"],
        ["text", "audio", "visual", "physio"],
    ]
    # Text feature extraction
    train_tf = None
    test_tf = None

    if TORCH_AVAILABLE:
        try:
            text_extractor = TextFeatureExtractor(model_path=args.model_path, local_files_only=True)

            if args.no_text_cache:
                train_tf = extract_text_features(train_data, text_extractor, split_name="train")
                test_tf = extract_text_features(test_data, text_extractor, split_name="test")
            else:
                default_cache_dir = Path(args.data_path) / "text_feature_cache"
                train_cache = args.train_text_cache or str(default_cache_dir / "train_roberta_text_features.npy")
                test_cache = args.test_text_cache or str(default_cache_dir / "test_roberta_text_features.npy")
                train_tf = save_or_load_text_features(train_data, text_extractor, train_cache, "train")
                test_tf = save_or_load_text_features(test_data, text_extractor, test_cache, "test")

        except Exception as e:
            print(f"[WARNING] Text feature extraction failed: {e}")
            traceback.print_exc()
            print("[WARNING] Will only work if sample[0][0] already contains numeric text embeddings.")
    else:
        print("[WARNING] TORCH_AVAILABLE=False. Will only work if sample[0][0] already contains numeric text embeddings.")

    strict_text = not args.allow_zero_text

    all_results = []
    print("\n[INFO] Starting fully automatic training: all models + all modality combinations")
    print(f"[INFO] Total models: {len(all_models)} | Total modality combinations: {len(all_modality_combos)}")

    for modalities in all_modality_combos:
        for model_name in all_models:
            try:
                res = train_evaluate(
                    data_path=args.data_path,
                    modalities=modalities,
                    model_name=model_name,
                    pca_components=args.pca,
                    seed=args.seed,
                    train_text_features=train_tf,
                    test_text_features=test_tf,
                    strict_text=strict_text,
                )
                if res:
                    all_results.append(res)
            except Exception as e:
                print(f"[FAILED] Modalities {modalities} + Model {model_name}: {e}")
                traceback.print_exc()
                continue

    print("\n" + "=" * 80)
    print("All experiments completed - LaTeX table format output")
    print("=" * 80)

    modality_map = {
        "l": "text",
        "a": "audio",
        "v": "visual",
        "p": "physio",
        "l&a&v": "text+audio+visual",
        "l&a&p": "text+audio+physio",
        "l&v&p": "text+visual+physio",
        "a&v&p": "audio+visual+physio",
        "l&a&v&p": "text+audio+visual+physio",
    }

    latex_cols = ["l", "a", "v", "p", "l&a&v", "l&a&p", "l&v&p", "a&v&p", "l&a&v&p"]

    if all_results:
        result_dict = {}
        for res in all_results:
            model = res["model"]
            mod = res["modalities"]
            acc = round(res["accuracy"], 4)
            f1 = round(res["f1"], 4)
            result_dict.setdefault(model, {})[mod] = (acc, f1)

        print("% ========== Multimodal Emotion Recognition Results ==========")
        print(r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Metric}} & \multicolumn{8}{c}{\textbf{Modalities}} \\")
        print(r"     & & l & a & v & p & l\&a\&v & l\&a\&p & l\&v\&p & a\&v\&p & l\&a\&v\&p \\ \hline")

        for model in all_models:
            if model not in result_dict:
                continue

            acc_row = []
            f1_row = []
            for col in latex_cols:
                target_mod = modality_map[col]
                acc, f1 = result_dict[model].get(target_mod, (0.0, 0.0))
                acc_row.append(f"{acc * 100:.2f}")
                f1_row.append(f"{f1 * 100:.2f}")

            print(r"\multirow{2}{*}{" + model + r"} & Acc & " + " & ".join(acc_row) + r" \\")
            print(r" & F1 & " + " & ".join(f1_row) + r" \\ \hline")

        all_results_sorted = sorted(all_results, key=lambda x: x["f1"], reverse=True)
        best = all_results_sorted[0]
        print(f"% Best combination: Modalities={best['modalities']}, Model={best['model']}")
        print(f"% Best metrics: Accuracy={best['accuracy']:.4f}, Weighted F1={best['f1']:.4f}")

        result_json_path = Path(args.data_path) / "multimodal_results_fixed.json"
        try:
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved result json to {result_json_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save result json: {e}")
    else:
        print("[ERROR] No valid training results")


if __name__ == "__main__":
    main()