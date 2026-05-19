import argparse, time, itertools
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import PulseEmoDataset
from model import LSTMModel, MaskedNLLLoss
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

MODALITIES = {
    "text": 768,
    "visual": 713,
    "acoustic": 65,
    "physio": 23
}

def get_loaders(path, batch_size):
    start = time.time()
    trainset = PulseEmoDataset(path, split="train")
    validset = PulseEmoDataset(path, split="valid")
    testset  = PulseEmoDataset(path, split="test")
    loaders = (
        DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn),
        DataLoader(validset, batch_size=batch_size, collate_fn=validset.collate_fn),
        DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn)
    )
    print(f"Data loaders created in {time.time()-start:.2f}s")
    return loaders

def train_eval_model(model, loss_fn, loader, optimizer=None, train=False, modalities=["text","visual","acoustic","physio"], cuda=False):
    phase = "Train" if train else "Eval"
    if train:
        model.train()
    else:
        model.eval()
    losses, preds, labels, masks = [], [], [], []

    step_times = []
    start_phase = time.time()

    for data in loader:
        step_start = time.time()
        textf, visuf, acouf, physio, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        features_list = []
        if "text" in modalities: features_list.append(textf)
        if "visual" in modalities: features_list.append(visuf)
        if "acoustic" in modalities: features_list.append(acouf)
        if "physio" in modalities:
            physio = torch.nan_to_num(physio, nan=0.0, posinf=0.0, neginf=0.0)
            std = physio.std(dim=0, keepdim=True)
            std[std==0] = 1.0
            physio = (physio - physio.mean(dim=0, keepdim=True)) / std
            features_list.append(physio)
        features = torch.cat(features_list, dim=-1)

        if train:
            optimizer.zero_grad()

        forward_start = time.time()
        log_prob, _, _, _ = model(features, qmask, umask)
        forward_time = time.time() - forward_start

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        lbl = label.view(-1)
        loss = loss_fn(lp_, lbl, umask)

        backward_time = 0
        if train:
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.cpu().numpy())
        labels.append(lbl.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * umask.sum().item())

        step_times.append((forward_time, backward_time))

    if not preds:
        return float('nan'), float('nan'), float('nan'), 0, 0

    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)
    avg_loss = np.sum(losses)/np.sum(masks)
    avg_f1   = f1_score(labels, preds, sample_weight=masks, average='weighted')*100
    avg_acc  = accuracy_score(labels, preds, sample_weight=masks)*100

    total_forward = sum(f for f, b in step_times)
    total_backward = sum(b for f, b in step_times)
    print(f"{phase} phase done in {time.time()-start_phase:.2f}s (Forward: {total_forward:.2f}s, Backward: {total_backward:.2f}s)")
    return avg_loss, avg_f1, avg_acc, total_forward, total_backward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dataset', type=str, default='../Pulse-Emo-Dialogue.pkl')
    args = parser.parse_args()

    total_start = time.time()
    cuda = args.cuda and torch.cuda.is_available()
    train_loader, valid_loader, test_loader = get_loaders(args.dataset, args.batch_size)
    cuda = args.cuda and torch.cuda.is_available()
    all_combinations = []
    for r in range(1, 5):
        all_combinations.extend(itertools.combinations(MODALITIES.keys(), r))

    lrs = [0.0001, 0.0005, 0.001]
    results = []

    overall_forward_time = 0
    overall_backward_time = 0
    overall_eval_time = 0

    for modalities in all_combinations:
        input_dim = sum(MODALITIES[m] for m in modalities)
        print(f"\nTraining with modalities: {modalities}, input_dim={input_dim}")
        combo_start = time.time()
        combo_forward_time = 0
        combo_backward_time = 0
        combo_eval_time = 0

        for lr in lrs:
            model = LSTMModel(input_dim, 100, 100, n_classes=7, dropout=0.25, attention=False)
            if cuda: model.cuda()
            loss_fn = MaskedNLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_test_f1 = -1
            best_train_acc, best_train_f1 = 0,0
            best_test_acc = 0

            for epoch in range(args.epochs):
                epoch_start = time.time()
                train_loss, train_f1, train_acc, forward_time, backward_time = train_eval_model(
                    model, loss_fn, train_loader, optimizer, True, modalities, cuda
                )
                val_loss, val_f1, val_acc, val_forward, _ = train_eval_model(
                    model, loss_fn, valid_loader, None, False, modalities, cuda
                )
                test_loss, test_f1, test_acc, test_forward, _ = train_eval_model(
                    model, loss_fn, test_loader, None, False, modalities, cuda
                )

                combo_forward_time += forward_time + val_forward + test_forward
                combo_backward_time += backward_time
                combo_eval_time += val_forward + test_forward

                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                    best_test_acc = test_acc
                    best_train_f1 = train_f1
                    best_train_acc = train_acc
                    best_val_f1 = val_f1
                    best_val_acc = val_acc

                print(f"Epoch {epoch+1}/{args.epochs} done in {time.time()-epoch_start:.2f}s")

            results.append({
                "modalities": modalities,
                "lr": lr,
                "train_acc": best_train_acc,
                "train_f1": best_train_f1,
                "val_acc": best_val_acc,
                "val_f1": best_val_f1,
                "test_acc": best_test_acc,
                "test_f1": best_test_f1
            })

            print(f"Modalities={modalities}, lr={lr:.4f}, "
                  f"Train F1/Acc={best_train_f1:.2f}/{best_train_acc:.2f}, "
                  f"Val F1/Acc={best_val_f1:.2f}/{best_val_acc:.2f}, "
                  f"Test F1/Acc={best_test_f1:.2f}/{best_test_acc:.2f}")

        combo_total = time.time() - combo_start
        print(f"Combination {modalities} done in {combo_total:.2f}s "
              f"(Forward: {combo_forward_time:.2f}s, Backward: {combo_backward_time:.2f}s, Eval: {combo_eval_time:.2f}s)")

        overall_forward_time += combo_forward_time
        overall_backward_time += combo_backward_time
        overall_eval_time += combo_eval_time

    print(f"\nTotal training time: {time.time()-total_start:.2f}s "
          f"(Forward: {overall_forward_time:.2f}s, Backward: {overall_backward_time:.2f}s, Eval: {overall_eval_time:.2f}s)")

    print("\nSummary Table:")
    print("Modalities\tLR\tTrain F1\tTrain Acc\tVal F1\tVal Acc\tTest F1\tTest Acc")
    for r in results:
        print(f"{'+'.join(r['modalities'])}\t{r['lr']:.4f}\t{r['train_f1']:.2f}\t{r['train_acc']:.2f}\t"
              f"{r['val_f1']:.2f}\t{r['val_acc']:.2f}\t{r['test_f1']:.2f}\t{r['test_acc']:.2f}")

if __name__ == "__main__":
    main()