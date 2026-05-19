import numpy as np, argparse, time, pickle, random, itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import PulseEmoDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, DialogRNNModel, DialogueGCNModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import builtins
import traceback

MODALITIES = {
    "text": 768,
    "visual": 713,
    "acoustic": 65,
    "physio": 23
}

_builtin_print = builtins.print
def my_print(*args, **kwargs):
    stack = traceback.extract_stack()
    if any("torch/utils/data" in frame.filename.replace("\\", "/") for frame in stack):
        return
    if any("train_plusEmo.py" in frame.filename for frame in stack):
        _builtin_print("[DEBUG PRINT]", *args, **kwargs)
    else:
        _builtin_print(*args, **kwargs)
builtins.print = my_print

seed = 100
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_PulseEmo_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = PulseEmoDataset(path=path, split="train")
    validset = PulseEmoDataset(path=path, split="valid")
    testset = PulseEmoDataset(path=path, split="test")

    train_loader = DataLoader(trainset,
                            batch_size=batch_size,
                            collate_fn=trainset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    valid_loader = DataLoader(validset,
                            batch_size=batch_size,
                            collate_fn=validset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                            batch_size=batch_size,
                            collate_fn=testset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, modalities,
                        cuda, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, physio, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        
        max_sequence_len.append(textf.size(0))

        feat_list = []
        if "text" in modalities:
            feat_list.append(textf)
        if "visual" in modalities:
            feat_list.append(visuf)
        if "acoustic" in modalities:
            feat_list.append(acouf)
        if "physio" in modalities:
            physio = torch.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)
            std = physio.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
            physio = (physio - physio.mean(dim=0, keepdim=True)) / std
            feat_list.append(physio)
        features = torch.cat(feat_list, dim=-1)

        log_prob, alpha, alpha_f, alpha_b, _ = model(features, qmask, umask)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, modalities,
                              cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, physio, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        feat_list = []
        if "text" in modalities:
            feat_list.append(textf)
        if "visual" in modalities:
            feat_list.append(visuf)
        if "acoustic" in modalities:
            feat_list.append(acouf)
        if "physio" in modalities:
            physio = torch.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)
            std = physio.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
            physio = (physio - physio.mean(dim=0, keepdim=True)) / std
            feat_list.append(physio)
        features = torch.cat(feat_list, dim=-1)

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        log_prob= model(features, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]

    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids

train_loader, valid_loader, test_loader = get_PulseEmo_loaders(
    '../Pulse-Emo-Dialogue.pkl', valid=0.0,
    batch_size=32, num_workers=0
)

if __name__ == '__main__':
    path = '../'
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')
    parser.add_argument('--graph-model', action='store_true', default=False, help='whether to use graph model after recurrent encoding')
    parser.add_argument('--nodal-attention', action='store_true', default=False, help='whether to use nodal attention in graph model')
    parser.add_argument('--windowp', type=int, default=10, help='context window size for past utterances')
    parser.add_argument('--windowf', type=int, default=10, help='context window size for future utterances')
    
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (base)')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.4, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.6, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print('Running on GPU' if args.cuda else 'Running on CPU')
    seed_everything()

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    n_classes  = 7
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    lr_candidates = [0.0001, 0.0005, 0.001, 0.002]
    modal_combinations = []
    for r in range(1, len(MODALITIES)+1):
        modal_combinations.extend(itertools.combinations(MODALITIES.keys(), r))
    
    all_experiments = []

    for modalities in modal_combinations:
        D_m = sum(MODALITIES[m] for m in modalities)
        print("\n" + "="*90)
        print(f"Experiment | Modalities: {' + '.join(modalities)} | Input Dim: {D_m} | Base Model: {args.base_model} | Graph Model: {args.graph_model}")
        print("="*90)

        for lr in lr_candidates:
            print(f"\n--- Learning Rate: {lr:.4f} ---")

            D_g = 150
            D_p = 150
            D_e = 100
            D_h = 100
            D_a = 100
            graph_h = 100

            if args.graph_model:
                model = DialogueGCNModel(
                    args.base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                    n_speakers=2, max_seq_len=110,
                    window_past=args.windowp, window_future=args.windowf,
                    n_classes=n_classes, listener_state=args.active_listener,
                    context_attention=args.attention, dropout=args.dropout,
                    nodal_attention=args.nodal_attention, no_cuda=args.no_cuda
                )
                name = 'Graph'
            else:
                if args.base_model == 'DialogRNN':
                    model = DialogRNNModel(
                        D_m, D_g, D_p, D_e, D_h, D_a, n_classes=n_classes,
                        listener_state=args.active_listener, context_attention=args.attention,
                        dropout_rec=args.rec_dropout, dropout=args.dropout
                    )
                elif args.base_model == 'GRU':
                    model = GRUModel(D_m, D_e, D_h, n_classes=n_classes, dropout=args.dropout)
                elif args.base_model == 'LSTM':
                    model = LSTMModel(D_m, D_e, D_h, n_classes=n_classes, dropout=args.dropout)
                else:
                    raise NotImplementedError("Base model must be DialogRNN/LSTM/GRU")
                name = 'Base'

            if cuda:
                model.cuda()

            counts = [1, 1, 1, 1, 1, 1, 1]
            weights = [1.0 / c for c in counts]
            weights = torch.FloatTensor(weights)
            weights = weights / weights.sum() * len(counts)
            loss_weights = weights

            if args.class_weight:
                if args.graph_model:
                    loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
                else:
                    loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                if args.graph_model:
                    loss_function = nn.NLLLoss()
                else:
                    loss_function = MaskedNLLLoss()

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)

            best_test_f1 = -1
            best_train_acc = best_train_f1 = 0
            best_val_acc = best_val_f1 = 0
            best_test_acc = 0
            all_fscore = []

            for e in range(n_epochs):
                start_time = time.time()

                if args.graph_model:
                    train_loss, train_acc, _, _, train_f1, _, _, _, _, _ = train_or_eval_graph_model(
                        model, loss_function, train_loader, e, modalities, cuda, optimizer, True
                    )
                    valid_loss, valid_acc, _, _, valid_f1, _, _, _, _, _ = train_or_eval_graph_model(
                        model, loss_function, valid_loader, e, modalities, cuda
                    )
                    test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                        model, loss_function, test_loader, e, modalities, cuda
                    )
                else:
                    train_loss, train_acc, _, _, _, train_f1, _ = train_or_eval_model(
                        model, loss_function, train_loader, e, modalities, cuda, optimizer, True
                    )
                    valid_loss, valid_acc, _, _, _, valid_f1, _ = train_or_eval_model(
                        model, loss_function, valid_loader, e, modalities, cuda
                    )
                    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(
                        model, loss_function, test_loader, e, modalities, cuda
                    )
                
                all_fscore.append(test_fscore)

                if test_fscore > best_test_f1:
                    best_test_f1 = test_fscore
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                    best_train_f1 = train_f1
                    best_val_acc = valid_acc
                    best_val_f1 = valid_f1

                if args.tensorboard:
                    writer.add_scalar(f'{"+".join(modalities)}/lr{lr}/test_acc', test_acc, e)
                    writer.add_scalar(f'{"+".join(modalities)}/lr{lr}/test_f1', test_fscore, e)

                print(f'Epoch {e+1:3d} | Train: Loss={train_loss:.4f} Acc={train_acc:.2f} F1={train_f1:.2f} | '
                      f'Val: Loss={valid_loss:.4f} Acc={valid_acc:.2f} F1={valid_f1:.2f} | '
                      f'Test: Loss={test_loss:.4f} Acc={test_acc:.2f} F1={test_fscore:.2f} | Time={round(time.time()-start_time,2)}s')

            all_experiments.append({
                "modalities": "+".join(modalities),
                "lr": lr,
                "input_dim": D_m,
                "base_model": args.base_model,
                "graph_model": args.graph_model,
                "train_acc": best_train_acc,
                "train_f1": best_train_f1,
                "val_acc": best_val_acc,
                "val_f1": best_val_f1,
                "test_acc": best_test_acc,
                "test_f1": best_test_f1,
                "max_test_f1": max(all_fscore)
            })
            print(f"\nBest for this config | Test F1={best_test_f1:.2f} (Max F1={max(all_fscore):.2f}) | Test Acc={best_test_acc:.2f}")

    print("\n\n" + "="*150)
    print("All Experiment Results Summary")
    print("="*150)
    header = f"{'Modal':<20} {'LR':<8} {'InDim':<8} {'Base':<10} {'Graph':<6} {'TrAcc':<8} {'TrF1':<8} {'VaAcc':<8} {'VaF1':<8} {'TeAcc':<8} {'TeF1':<8}"
    print(header)
    print("-"*150)
    for exp in all_experiments:
        print(f"{exp['modalities']:<20} {exp['lr']:.4f}     {exp['input_dim']:<8} {exp['base_model']:<10} {exp['graph_model']:<6} "
              f"{exp['train_acc']:<8.2f} {exp['train_f1']:<8.2f} {exp['val_acc']:<8.2f} {exp['val_f1']:<8.2f} "
              f"{exp['test_acc']:<8.2f} {exp['test_f1']:<8.2f}")

    best_exp = max(all_experiments, key=lambda x: x['test_f1'])
    print("\nBest Overall Result")
    print(f"Modalities: {best_exp['modalities']} | LR: {best_exp['lr']:.4f} | Input Dim: {best_exp['input_dim']}")
    print(f"Base Model: {best_exp['base_model']} | Graph Model: {best_exp['graph_model']}")
    print(f"Test F1: {best_exp['test_f1']:.2f} | Test Acc: {best_exp['test_acc']:.2f}")
    print(f"Val F1: {best_exp['val_f1']:.2f} | Val Acc: {best_exp['val_acc']:.2f}")

    if args.tensorboard:
        writer.close()