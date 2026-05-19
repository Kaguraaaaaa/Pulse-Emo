import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from create_dataset import MOSI, MOSEI, UR_FUNNY, PULSE_EMO, PAD, UNK

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


def _build_roberta_tokenizer():
    """
    RoBERTa tokenizer is only used for legacy SATI models.
    Keep it lazy so Qwen training does not depend on local RoBERTa files.
    """
    vocab_file = '/data3/xgy/SATI/model/vocab.json'
    merges_file = '/data3/xgy/SATI/model/merges.txt'
    return RobertaTokenizer(vocab_file, merges_file)


class MSADataset(Dataset):
    def __init__(self, config):

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        elif "pulse_emo" in str(config.data_dir).lower():
            dataset = PULSE_EMO(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        # 如果启用了合并验证集和测试集，且当前模式是test，则合并dev和test数据
        if hasattr(config, 'merge_dev_test') and config.merge_dev_test and config.mode == 'test':
            dev_data, _, _ = dataset.get_data('dev')
            test_data, word2id, pretrained_emb = dataset.get_data('test')
            self.data = dev_data + test_data
            self.word2id = word2id
            self.pretrained_emb = pretrained_emb
            print(f"合并验证集和测试集: dev={len(dev_data)}, test={len(test_data)}, 总计={len(self.data)}")
        else:
            self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        #print(self.data)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]
        
        # 检查是否有physio模态（数据格式长度为5表示有physio）
        if len(self.data[0][0]) == 5:
            config.physio_size = self.data[0][0][3].shape[1]
        else:
            config.physio_size = 0  # 无physio特征

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb


    def __getitem__(self, index):
        sample = self.data[index]
        
        # 处理PULSE_EMO数据格式：如果有5个特征，包含physio
        if len(sample[0]) == 5:
            words, visual, acoustic, physio, actual_words = sample[0]
            # 返回所有特征包括physio
            modified_sample = ((words, visual, acoustic, physio, actual_words), sample[1])
            if len(sample) > 2:  # 如果有第三个元素（segment信息）
                modified_sample = modified_sample + (sample[2],)
            return modified_sample
        else:
            return sample

    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    #print(config.mode)
    config.data_len = len(dataset)

    # Build tokenizer once per dataloader.
    if getattr(config, "llm_type", "qwen").lower() == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(
            getattr(config, "llm_path", "/data/s22xjq/Qwen"),
            trust_remote_code=True,
        )
        # Some LLM tokenizers (especially local/custom) may not define pad_token.
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = _build_roberta_tokenizer()


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things

        # Choose tensor type based on dataset type
        if "ur_funny" in str(config.data_dir).lower() or "pulse_emo" in str(config.data_dir).lower():
            # Classification task - use LongTensor for labels
            processed_labels = []
            for sample in batch:
                label = sample[1]
                if isinstance(label, np.ndarray):
                    label = int(label.flatten()[0])  # 取第一个元素并转为int
                else:
                    label = int(label)
                processed_labels.append(label)
            labels = torch.LongTensor(processed_labels)
        else:
            # Regression task (mosi/mosei) - use FloatTensor for labels
            labels = torch.cat([torch.FloatTensor([sample[1]]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)#[44, 64]
        #print(sentences.shape)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])  #[44, 64, 47]
        #print(visual)
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch]) #[44, 64, 47]
        
        # 处理physio模态（如果存在）
        if len(batch[0][0]) == 5:  # 有physio特征
            physio = pad_sequence([torch.FloatTensor(sample[0][3]) for sample in batch])
        else:
            physio = None
        #print(sentences.shape, visual.shape)

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer
        roberta_details = []
        for sample in batch:
            # 获取实际的文本词汇
            if len(sample[0]) == 5:  # PULSE_EMO格式: (words, visual, acoustic, physio, actual_words)
                actual_words = sample[0][4]  # 最后一个是actual_words
            else:  # 其他数据集格式: (words, visual, acoustic, actual_words)
                actual_words = sample[0][3]
            
            # 确保actual_words是字符串列表
            if isinstance(actual_words, list) and len(actual_words) > 0:
                text = " ".join(str(word) for word in actual_words)
            else:
                text = ""  # 空文本的fallback
            
            # 限制最大长度，避免过长的序列导致问题
            max_len = min(SENT_LEN, int(getattr(config, "max_text_len", 512)))
            encoded_roberta_sent = tokenizer.encode_plus(
                text, 
                max_length=max_len,
                add_special_tokens=True, 
                padding='max_length',  # 填充到最大长度
                truncation=True,       # 截断超过最大长度的部分
                return_tensors='pt'    # 返回PyTorch tensors（如果你使用的是PyTorch）
            )
            roberta_details.append(encoded_roberta_sent)

        # 在循环外部创建BERT张量
        bert_sentences = torch.LongTensor([sample["input_ids"].squeeze(0).tolist() for sample in roberta_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"].squeeze(0).tolist() for sample in roberta_details])
        bert_sentence_types = torch.randn(0)
        
        # 确保BERT序列长度与原始序列长度匹配
        if bert_sentences.shape[1] != SENT_LEN:
            # 如果BERT序列长度与原始序列长度不匹配，需要调整
            if bert_sentences.shape[1] > SENT_LEN:
                # 截断BERT序列
                bert_sentences = bert_sentences[:, :SENT_LEN]
                bert_sentence_att_mask = bert_sentence_att_mask[:, :SENT_LEN]
            else:
                # 填充BERT序列
                pad_length = SENT_LEN - bert_sentences.shape[1]
                bert_sentences = torch.cat([
                    bert_sentences, 
                    torch.zeros(bert_sentences.shape[0], pad_length, dtype=bert_sentences.dtype)
                ], dim=1)
                bert_sentence_att_mask = torch.cat([
                    bert_sentence_att_mask, 
                    torch.zeros(bert_sentence_att_mask.shape[0], pad_length, dtype=bert_sentence_att_mask.dtype)
                ], dim=1)
        
        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        # 返回physio如果存在
        if physio is not None:
            return sentences, visual, acoustic, physio, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask
        else:
            return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=False)  # 保留最后一个批次，但需要在collate_fn中处理

    return data_loader
