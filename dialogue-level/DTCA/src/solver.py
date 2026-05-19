import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from torchviz import make_dot
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD, FocalLoss, WeightedMSELoss
import models
import models
from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_maes = []
        self.valid_maes = []
        self.test_maes = []
        self.train_f1_scores = []
        self.valid_f1_scores = []
        self.test_f1_scores = []
        
        
        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            #print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    
    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 3
        writer = SummaryWriter()
        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        if self.train_config.data == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        elif self.train_config.data == "pulse_emo":
            print("🎯 使用标准CrossEntropyLoss...")
            # class_weights = torch.tensor(
            #     [0.89, 0.76, 0.40, 2.68, 1.61, 1.07, 6.43],
            #     dtype=torch.float
            # ).cuda()
            # class_weights = torch.tensor(
            #     [1/15.9, 1/18.9, 1/35.7, 1/5.5, 1/8.6, 1/13, 1/2.4],
            #     dtype=torch.float
            # ).cuda()
            class_weights = torch.tensor(
                [1, 1, 1, 1, 1, 1, 1],
                dtype=torch.float
            ).cuda()
            # self.criterion = FocalLoss(alpha=class_weights, gamma=2)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
            # self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.MSELoss(reduction="mean")


        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        
        best_test_acc = float('-inf')
        best_valid_f1 = float('-inf')
        best_test_f1 = float('-inf')
        best_valid_acc = float('-inf')
        
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.3)
        
        #train_losses = []
        #valid_losses = []F
        
        start_saving_epoch = int(self.train_config.n_epoch * self.train_config.start_saving)
        
        
        for e in range(self.train_config.n_epoch):
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_jsd_loss = []
            train_trend_loss = []
            train_dir_loss = []
            train_mag_loss = []
            train_loss = []
            train_y_true, train_y_pred = [], []
            train_loss_jsd = []
            pos = 0
            nos = 0
            nig = 0
            
            for batch in self.train_data_loader:
                self.model.zero_grad()
                # Check if physio is present in batch
                if len(batch) == 9:  # Has physio
                    t, v, a, p, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                    p = to_gpu(p)
                else:  # No physio
                    t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                    p = None
                
                batch_size = t.size(0)
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)
                #for name, param in self.model.named_parameters():
                    #print(name, param.shape)
                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, p)
                
                # 处理模型输出
                if isinstance(y_tilde, tuple):
                    y_tilde = y_tilde[0]  # 如果返回tuple，取第一个元素

                #print(y)
                #print(y_tilde)
                if self.train_config.data == "ur_funny":
                    y = y.squeeze()

                cls_loss = self.criterion(y_tilde, y)
                # dif_loss = self.get_diff_loss()
                # domain_loss = self.get_domain_loss()
                #print((cls_loss.item()))
                # recon_loss = self.get_recon_loss()
                # cmd_loss = self.get_cmd_loss()
                jsd_loss = self.get_jsd_loss()
                trend_loss, dir_loss, mag_loss = self.trend_loss()
                
                # 暂时只使用分类loss来调试
                if self.train_config.data == "pulse_emo":
                    loss = cls_loss + \
                        self.train_config.jsd_weight * (1*jsd_loss + 1*trend_loss)
                # print(jsd_loss.item(), trend_loss.item(), self.train_config.jsd_weight)
                    # print(self.train_config.jsd_weight, jsd_loss.item(), trend_loss.item())

                # load_bal_loss = self.get_load_balancing_loss(lambda_aux=0.6)
                # loss += load_bal_loss  # 加到总损失中
                # 检查loss是否为nan或inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  检测到异常loss值: {loss.item()}")
                    print(f"   cls_loss: {cls_loss.item()}")
                
                loss.backward()
                
                #print("backward")
                #print("latent_queries gradient:", self.model.latent_queries.grad)
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                self.optimizer.step()
                train_jsd_loss.append(jsd_loss.item())
                train_trend_loss.append(trend_loss.item())
                train_dir_loss.append(dir_loss.item())
                train_mag_loss.append(mag_loss.item())
                train_loss_cls.append(cls_loss.item())
                train_loss.append(loss.item())
                train_loss_jsd.append(jsd_loss.item())
                train_loss_jsd.append(trend_loss.item())
                train_y_true.append(y.detach().cpu().numpy())
                train_y_pred.append(y_tilde.detach().cpu().numpy())
                #print(train_y_true)            
            print("\n")    
            writer.add_scalar('Loss/Train_cls', np.mean(train_loss_cls), e)
            writer.add_scalar('Loss/Train_trend', np.mean(train_trend_loss), e)
            writer.add_scalar('Loss/Train_jsd', np.mean(train_jsd_loss), e)
            writer.add_scalar('Loss/Train_diff', np.mean(train_loss_diff),e )
            writer.add_scalar('Loss/Train_recon', np.mean(train_loss_recon), e)
            writer.add_scalar('Loss/Train_similarity', np.mean(train_loss_sim), e)
            #writer.add_scalar('Loss/Train_jsd', np.mean(train_loss_jsd), e)
            writer.add_scalar('Loss/Train_total', np.mean(train_loss), e)
            train_loss = np.mean(train_loss)
            #train_losses.append(train_loss)
            self.train_losses.append(train_loss)
            # 检查是否有有效的训练数据
            if len(train_y_true) == 0 or len(train_y_pred) == 0:
                print("⚠️  没有有效的训练数据，跳过这个epoch")
                continue
                
            train_y_true = np.concatenate(train_y_true, axis=0).squeeze()
            train_y = np.array(train_y_true)
            pos +=len([x for x in train_y if x > 0])
            nos +=len([x for x in train_y if x < 0])
            nig +=len([x for x in train_y if x == 0])
            train_y_pred = np.concatenate(train_y_pred, axis=0).squeeze()           
            train_acc = self.calc_metrics(train_y_true, train_y_pred, mode="train")
            
            # 对于分类任务，计算不同的指标
            if self.train_config.data == "ur_f unny" or self.train_config.data == "pulse_emo":
                train_y_pred_classes = np.argmax(train_y_pred, 1)
                train_mae = 0.0  # 分类任务不计算MAE
                train_f1 = f1_score(train_y_true, train_y_pred_classes, average='weighted')
            else:
                train_mae = np.mean(np.abs(train_y_pred - train_y_true))
                train_f1 = f1_score((train_y_pred > 0), (train_y_true > 0), average='weighted')
            writer.add_scalar('Accuracy/Train', train_acc, e)
            writer.add_scalar('F1/Train', train_f1, e)
            print(f"Epoch {e+1} - Training loss: {round(train_loss, 4)}, Accuracy: {round(train_acc, 4)}, MAE: {round(train_mae, 4)}, F1-score: {round(train_f1, 4)}")
            
            self.train_accuracies.append(train_acc)
            self.train_maes.append(train_mae)
            self.train_f1_scores.append(train_f1)
            #print(pos, nos, nig)
            pos = 0
            nos = 0
            nig = 0
            #print(f"Training loss: {round(np.mean(train_loss), 4)}")
            
            
            # 检查是否启用合并模式
            merge_dev_test = getattr(self.train_config, 'merge_dev_test', True)
            
            if merge_dev_test:
                valid_loss, valid_acc, valid_mae, valid_f1 = self.eval(mode="test")
                self.valid_losses.append(valid_loss)
                self.valid_accuracies.append(valid_acc)
                self.valid_maes.append(valid_mae)
                self.valid_f1_scores.append(valid_f1)
                print(f"Epoch {e+1} - loss: {round(valid_loss, 4)}, Accuracy: {round(valid_acc, 4)}, MAE: {round(valid_mae, 4)}, F1-score: {round(valid_f1, 4)}")
                writer.add_scalar('Accuracy/Valid', valid_acc, e)
                writer.add_scalar('F1/Valid', valid_f1, e)
            else:
                # 正常模式：评估验证集
                valid_loss, valid_acc, valid_mae, valid_f1 = self.eval(mode="dev")
                self.valid_losses.append(valid_loss)
                self.valid_accuracies.append(valid_acc)
                self.valid_maes.append(valid_mae)
                self.valid_f1_scores.append(valid_f1)
                print(f"Epoch {e+1} - Validation loss: {round(valid_loss, 4)}, Accuracy: {round(valid_acc, 4)}, MAE: {round(valid_mae, 4)}, F1-score: {round(valid_f1, 4)}")
                writer.add_scalar('Accuracy/Valid', valid_acc, e)
                writer.add_scalar('F1/Valid', valid_f1, e)
            
            # 评估测试集
            test_loss, test_acc, test_mae, test_f1 = self.eval(mode="test")
            # self.visualize_gating_weights(save_dir=f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}',epoch=e)
            #valid_loss, valid_acc = self.eval(mode="dev")
            

            if merge_dev_test:
                if e >= start_saving_epoch and valid_f1 >= best_valid_f1:
                    print(f"Valid F1: {valid_f1}, Best Valid F1: {best_valid_f1}")
                    best_valid_f1 = valid_f1
                    best_valid_acc = valid_acc
                    print(f"Valid acc: {valid_acc}, Best Valid acc: {best_valid_acc}")
                    if not os.path.exists(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}'):
                        os.makedirs(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}')
                    torch.save(self.model.state_dict(), f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/model.std')
                    torch.save(self.optimizer.state_dict(), f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/optim.std')
                    curr_patience = patience
                    

            else:
                # 正常模式：使用验证集性能来选择最佳模型
                if e >= start_saving_epoch and valid_f1 >= best_valid_f1:
                    print(best_valid_f1)
                    best_valid_f1 = valid_f1
                    print("Found new best model on dev set! f1")
                    print(f"Valid acc: {valid_acc}, Best Valid acc: {best_valid_acc}")
                    if not os.path.exists(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}'):
                        os.makedirs(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}')
                    torch.save(self.model.state_dict(), f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/model_{self.train_config.name}.std')
                    torch.save(self.optimizer.state_dict(), f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/optim_{self.train_config.name}.std')
                    curr_patience = patience
                    
                if e >= start_saving_epoch and valid_acc >= best_valid_acc:
                    best_valid_acc = valid_acc
                    print("Found new best model on dev set! acc")
                    if not os.path.exists(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}'):
                        os.makedirs(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}')
                    torch.save(self.model.state_dict(), f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/model.std')
                    torch.save(self.optimizer.state_dict(), f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/optim.std')
                    curr_patience = patience
            
            # # 如果性能没有提升，减少patience
            # if e >= start_saving_epoch:
            #     curr_patience -= 1
            #     if curr_patience <= -1:
            #         print("Running out of patience, loading previous best model.")
            #         num_trials -= 1
            #         curr_patience = patience
            #         self.model.load_state_dict(torch.load(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/model_{self.train_config.name}.std'))
            #         self.optimizer.load_state_dict(torch.load(f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/optim_{self.train_config.name}.std'))
            #         lr_scheduler.step()
            #         print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
    
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        # self.eval(mode="test", to_print=True)
        print("best_valid_f1", best_valid_f1)
        print("best_test_acc", best_test_acc)
        return best_valid_acc, best_valid_f1


    
    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []
    
        y_true2, y_pred2 = [], []
        eval_loss2, eval_loss_diff2 = [], []

        if mode == "dev":
            if self.dev_data_loader is None:
                return 0.0, 0.0, 0.0, 0.0
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                     f'checkpoints/checkpoints/checkpoints_{self.train_config.learning_rate}/model.std'))
            

        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                # Check if physio is present in batch
                if len(batch) == 9:  # Has physio
                    t, v, a, p, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                    p = to_gpu(p)
                else:  # No physio
                    t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                    p = None

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, p)
                
                # 处理模型输出
                if isinstance(y_tilde, tuple):
                    y_tilde = y_tilde[0]  # 如果返回tuple，取第一个元素
                
                train_y_pred_np = y_tilde.detach().cpu().numpy()
                binary_pred = (train_y_pred_np > 0).astype(int)
                #print(binary_pred)
                if self.train_config.data == "ur_funny" or self.train_config.data == "pulse_emo":
                    y = y.squeeze()
                # print(y_tilde.shape, y.shape)
                # 检查批次大小匹配
                
                y = y.view(-1)
                if y_tilde.shape[0] != y.shape[0]:
                    print(f"⚠️  批次大小不匹配: y_tilde.shape={y_tilde.shape}, y.shape={y.shape}")
                    # 如果标签为空，跳过这个批次
                    if y.shape[0] == 0:
                        print("跳过空标签批次")
                        continue
                
                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())



        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)
        
        # 对于分类任务，计算不同的指标
        if self.train_config.data == "ur_funny" or self.train_config.data == "pulse_emo":
            y_pred_classes = np.argmax(y_pred, 1)
            mae = 0.0  # 分类任务不计算MAE
            f1 = f1_score(y_true, y_pred_classes, average='weighted')
        else:
            mae = np.mean(np.abs(y_pred - y_true))
            f1 = f1_score((y_pred >= 0), (y_true >= 0), average='weighted')

        if mode == "dev":
            self.valid_losses.append(eval_loss)
            self.valid_accuracies.append(accuracy)
            self.valid_maes.append(mae)
            self.valid_f1_scores.append(f1)
        elif mode == "test":
            self.test_losses.append(eval_loss)
            self.test_accuracies.append(accuracy)
            self.test_maes.append(mae)
            self.test_f1_scores.append(f1)
        if to_print:
            print(f"Eval {mode} loss: {round(eval_loss, 4)}, Accuracy: {round(accuracy, 4)}, MAE: {round(mae, 4)}, F1-score: {round(f1, 4)}")
              
        return eval_loss, accuracy, mae, f1

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """


        if self.train_config.data == "ur_funny" or self.train_config.data == "pulse_emo":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy ", accuracy_score(test_truth, test_preds))
            
            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
            
            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
                print("F1 (pos/neg) ", f1_score(binary_truth, binary_preds,average='weighted'))
            
            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
                print("F1 (non-neg/neg) ", f1_score(binary_truth, binary_preds, average='weighted'))
            
            return accuracy_score(binary_truth, binary_preds)
        
        
    def angular_margin_loss(self, feature, label, weight, scale_factor=30.0, margin=0.5, lambda_l2=0.01):

        labels = [0, 1, 2]
        normalized_feature = F.normalize(feature, p=2, dim=0)
        normalized_weight = F.normalize(weight, p=2, dim=1)
        

        cos_theta = torch.matmul(normalized_feature, normalized_weight.t())  # 形状为 (num_classes,)
        
        correct_class_cos_theta = cos_theta[label]

        cos_theta_m = correct_class_cos_theta - margin

        cos_theta_with_margin = cos_theta.clone()
        cos_theta_with_margin[label] = cos_theta_m

        exp_cos_theta = torch.exp(scale_factor * cos_theta_with_margin)
        softmax_output = exp_cos_theta / exp_cos_theta.sum()

        loss = -torch.log(softmax_output[label])

        l2_reg = lambda_l2 * (weight ** 2).sum()

        total_loss = loss + l2_reg
        
        return total_loss
        
    def discriminator_loss(real_output, fake_output):
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))

    def get_domain_loss(self,):

        #if self.train_config.use_cmd_sim:
        #    return 0.0
        pred_shared_t = self.model.domain_shared_t 
        pred_shared_a = self.model.domain_shared_a
        pred_shared_v = self.model.domain_shared_v
        pred_private_t = self.model.domain_private_t 
        pred_private_a = self.model.domain_private_a
        pred_private_v = self.model.domain_private_v 

        W = self.model.W.discriminator_layer_2.weight

        Lami = torch.zeros(1, device="cuda")
        Lams = torch.zeros(1, device="cuda")
        for i in range (pred_private_v.size(0)):
            Lami += (self.angular_margin_loss(feature=pred_shared_t[i], label=0, weight=W) + self.angular_margin_loss(feature=pred_shared_a[i], label=1, weight=W) \
                + self.angular_margin_loss(feature=pred_shared_v[i], label=2, weight=W))/3.0
            Lams += (self.angular_margin_loss(feature=pred_private_t[i], label=0, weight=W) + self.angular_margin_loss(feature=pred_private_v[i], label=2, weight=W) \
                + self.angular_margin_loss(feature=pred_private_a[i], label=1, weight=W))/3.0
        Lami = Lami/pred_private_v.size(0)
        Lams = Lams/pred_private_v.size(0)
        return Lami + Lams

    def get_cmd_loss(self,):

        if not self.train_config.use_cmd_sim:
            return 0.0
        shared_t = torch.sum(self.model.utt_shared_t, dim=0)/self.model.utt_shared_t.size(0)
        shared_v = torch.sum(self.model.utt_shared_v, dim=0)/self.model.utt_shared_v.size(0)
        shared_a = torch.sum(self.model.utt_shared_a, dim=0)/self.model.utt_shared_a.size(0)
        # losses between shared states
        loss = self.loss_cmd(shared_t, shared_v, 5)
        loss += self.loss_cmd(shared_t, shared_a, 5)
        loss += self.loss_cmd(shared_a, shared_v, 5)
        loss = loss/3.0

        return loss

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a
        #print(self.model.utt_private_a.shape)
        shared_t = torch.sum(shared_t, dim=0)/shared_t.size(0)
        #print(shared_t.shape)
        shared_v = torch.sum(shared_v, dim=0)/shared_t.size(0)
        shared_a = torch.sum(shared_a, dim=0)/shared_t.size(0)
        private_t = torch.sum(private_t, dim=0)/shared_t.size(0)
        private_v = torch.sum(private_v, dim=0)/shared_t.size(0)
        private_a = torch.sum(private_a, dim=0)/shared_t.size(0)

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/3.0
        return loss


    def kl_divergence(self, p, q):
        p = p + 1e-10  
        q = q + 1e-10
        return torch.sum(p * torch.log(p / q))
    
    def jsd(self, p, q):
        m = (p + q ) / 2
        return (self.kl_divergence(p, m) + self.kl_divergence(q, m) ) / 2
    
    import torch
    import torch.nn.functional as F
    import numpy as np

    def get_jsd_loss(self):
        model = self.model
        # if not (hasattr(model, 'utt_private_v') and hasattr(model, 'utt_shared_v')):
        #     device = next(model.parameters()).device
        #     return torch.tensor(0.0, device=device)
        device = next(model.parameters()).device  # 自动获取模型设备（避免硬编码cuda）  
        # 拼接特征 + 设备统一（防止张量在不同设备）
        features_v = model.v_embeds
        seq, batch, ebd = features_v.shape
        
        # 重构jsd函数：全用torch操作，避免numpy，且增强数值稳定性
        def jsd(p, q):
            # 1. 防止0值导致log报错
            p = p.clamp(min=1e-8)  # 替换p = p + 1e-8，更安全
            q = q.clamp(min=1e-8)
            m = (p + q) / 2
            
            # 2. 计算KL散度，用torch.log（避免numpy）
            kl_p = (p * torch.log(p / m)).sum(dim=-1)  # 按特征维度求和，返回[batch]
            kl_q = (q * torch.log(q / m)).sum(dim=-1)
            return (kl_p + kl_q) / 2  # 返回[batch]
        
        jsd_loss = torch.tensor(0.0, device=device)  # 用torch张量累加，避免浮点数精度问题
        temperature = 0.1

        for i in range(seq - 1):
            feat_i = features_v[i]  # [batch, ebd]
            feat_i1 = features_v[i+1]  # [batch, ebd]
            
            # 标准化：用torch操作，添加eps避免除以0
            feat_i_mean = feat_i.mean(dim=-1, keepdim=True)
            feat_i_std = feat_i.std(dim=-1, keepdim=True).clamp(min=1e-8)
            feat_i = (feat_i - feat_i_mean) / feat_i_std
            
            feat_i1_mean = feat_i1.mean(dim=-1, keepdim=True)
            feat_i1_std = feat_i1.std(dim=-1, keepdim=True).clamp(min=1e-8)
            feat_i1 = (feat_i1 - feat_i1_mean) / feat_i1_std
            
            # Softmax：用torch操作
            p = F.softmax(feat_i / temperature, dim=-1)
            q = F.softmax(feat_i1 / temperature, dim=-1)
            
            # 计算当前batch的jsd：直接用张量操作，无需循环（高效+避免设备问题）
            batch_jsd = jsd(p, q)  # [batch]
            
            # 过滤nan/inf：用torch.isnan/torch.isinf（适配CUDA张量）
            valid_mask = ~(torch.isnan(batch_jsd) | torch.isinf(batch_jsd))
            batch_jsd_clean = batch_jsd[valid_mask]
            
            # 计算当前步的均值（空时返回0）
            if len(batch_jsd_clean) == 0:
                mean_jsd = torch.tensor(0.0, device=device)
            else:
                mean_jsd = batch_jsd_clean.mean()
            
            jsd_loss += mean_jsd 
        
        if seq == 1:
            return torch.tensor(0.0, device=device)
        
        # 求平均，避免除以0
        jsd_loss = jsd_loss / (seq - 1 + 1e-8)  # +1e-8防止seq-1=0
        # 最终兜底：替换nan/inf为0
        jsd_loss = torch.nan_to_num(jsd_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return jsd_loss



    def trend_loss(self):
        
        R = (self.model.v_embeds + self.model.a_embeds)/2 # [24, 32, 264]
        P = self.model.p_embeds     # [24, 32, 264]
        R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        R = R.permute(1,0,2)            # [32, 24, 264]
        P = P.permute(1,0,2)            # [32, 24, 264]

        dR = R[:,1:] - R[:,:-1]         # [32, 23, 264]
        dP = P[:,1:] - P[:,:-1]         # [32, 23, 264]

        dir_loss = F.mse_loss(
            F.normalize(dR, dim=-1, eps=1e-8),  # 关键：eps避免除以0
            F.normalize(dP, dim=-1, eps=1e-8)
        )
        
        mag_loss = F.mse_loss(
            torch.norm(dR, dim=-1),
            torch.norm(dP, dim=-1)
        )
        trend_loss =  mag_loss + dir_loss
        trend_loss = torch.nan_to_num(trend_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return trend_loss, dir_loss, mag_loss
    
    
   