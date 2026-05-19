import numpy as np
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
path = "./PulseEmo/PulseEmo.pkl"
with open(path, 'rb') as f:
    data = pickle.load(f)

sentences = data["Sentences"]
speakers = data["Speakers"]
subject_labels = data["SubjectLabels"]
object_labels = data["ObjectLabels"]
dialog_ids = data["Dialog"]
textual = np.array(data["Textual"])    # 形状: (n_samples, textual_dim 768)
acoustic = np.array(data["Acoustic"])  # 形状: (n_samples, acoustic_dim 65)
visual = np.array(data["Visual"])      # 形状: (n_samples, visual_dim 713)
vid_split = data["Vid"]

import numpy as np
from sklearn.model_selection import train_test_split

# 获取标签数组
labels = np.array(object_labels)

# 第一步：分出训练和剩余（验证+测试）
train_idx, temp_idx = train_test_split(
    np.arange(len(labels)),
    test_size=0.3,
    stratify=labels,
    random_state=42
)

# 第二步：将剩余再分成验证和测试
valid_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=labels[temp_idx],
    random_state=42
)

# 创建新的 split 标签
vid_split = np.empty(len(labels), dtype=object)
vid_split[train_idx] = "train"
vid_split[valid_idx] = "valid"
vid_split[test_idx] = "test"

# 替换原数据结构
data["Vid"] = vid_split.tolist()

# 展示划分比例
total = len(labels)
def show_ratio(name, idx):
    count = len(idx)
    percent = count / total * 100
    label_dist = Counter(labels[idx])
    for key, val in label_dist.items():
        label_dist[key] = val/count
    print(f"{name}: {count} samples ({percent:.1f}%) -> label分布: {dict(label_dist)}")

show_ratio("Train", train_idx)
show_ratio("Valid", valid_idx)
show_ratio("Test", test_idx)

# 保存为新文件
with open("PulseEmo.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ 已根据 object_labels 重新划分并保存为 'new_split_data.pkl'")