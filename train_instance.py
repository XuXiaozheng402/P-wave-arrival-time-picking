
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
# from utils.data import DitingData, DitingDataThread
from models.DAMS import UNet_mpt
from models.BRNN_instance import BRNN
from models.unetpp_instance import UNetpp
from models.UNetAS_instance import UNet
# from models.EQT import EQTransformer
from models.EQT_instance import EQTransformer
# from models.EQTransformer_instance import EQTransformer
# from models.UNet import UNet, UNet_mpt
# from models.UNETitself import UNet_mpt,UNet
# from models.UNetDA import UNet_mpt
from models.phasenet_instance import PhaseNet
# from models.UNet_v2 import UNetV2, UNet_mptV2
# from mydataset.seismic_dataset import SeismicDataset
from mydataset.seismic_dataset_instance import SeismicDataset

from mydataset.seismic_dataset_v2 import SeismicDatasetV2
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import random
from test import plot_waveform
from datetime import datetime, timedelta
import os
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, title=True):
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind, x[ind]


def extract_picks(
    preds,
):
    mph = {}

    mph["P"] = 0.3
    mph["S"] = 0.3
    mph["PS"] = 0.3
    mpd = 50
    Nb, Nt, Nc = preds.shape

    picks = []
    for i in range(Nb):
        p_picks, s_picks = [], []
        idxs, probs = detect_peaks(preds[i, :, 0], mph=mph['P'], mpd=mpd, show=False)
        idxs = idxs.tolist()

        p_picks += idxs

        picks.append(idxs)

    return picks


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    fix_random_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    csv_file_path = "D:/X/p_wave/splits_csv/split_0.csv"


    train_dataset = SeismicDataset(args.input,csv_file=csv_file_path, start=0, end=40000, phase='train')
    valid_dataset = SeismicDataset(args.input,csv_file=csv_file_path, start=40001, end=62000, phase='test')

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f'Train data size: {len(train_dataset)}, val data size: {len(valid_dataset)}')

    # lossfn = torch.nn.CrossEntropyLoss()
    lossfn = torch.nn.BCEWithLogitsLoss()
    if args.model == 'unet':
        model = UNet() if args.type == 'v1' else UNet()
        # 在 main 函数中创建 UNet 实例时传递参数
        # model = UNet(in_channels=38, out_channels=1) if args.type == 'v1' else UNet(in_channels=38, out_channels=1)

    elif args.model == 'unet_mpt':
        model = UNet_mpt() if args.type == 'v1' else UNet_mpt()
    elif args.model == 'EQTransformer':
        model =EQTransformer() if args.type == 'v1' else EQTransformer()
    elif args.model == 'unetpp':
        model =UNetpp() if args.type == 'v1' else UNetpp()
    elif args.model == 'phasenet':
        model =PhaseNet() if args.type == 'v1' else PhaseNet()
    elif args.model == 'rnn':
        model =BRNN() if args.type == 'v1' else BRNN()
    else:
        raise ValueError
    model.to(device)
    lossfn.to(device)
    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0e-4)
    best_f1 = 0

    for epoch in range(50):
        model.train()
        for step, batch in enumerate(train_dataloader):
            sample, label, raw_sample, fname = batch
            # 在模型和数据处理代码中
            sample = sample.to(torch.float32)

            sample = sample.to(device).permute(0, 2, 1)
            label = label.to(device).permute(0, 2, 1)
            ##########phasenet模型加这句话
            # label = label.expand(-1, 3, -1)
            sample = sample.float()
            pred = model(sample)
            # print("Sample shape:", sample.shape)
            # print("Label sample:", label[0, :, 0])  # 打印第一个样本的第一个通道的标签

            loss = lossfn(pred, label)
            loss.backward()

            optim.step()
            optim.zero_grad()

            if step % 300 == 0:
                print('loss: ', loss.item())

            # plot
            if args.plot:
                pred = torch.sigmoid(pred)
                pred = pred.permute(0, 2, 1).cpu().detach().clone().numpy()
                label = label.permute(0, 2, 1).cpu().numpy()

                pred_picks = extract_picks(preds=pred)
                label_picks = extract_picks(preds=label)
                raw_sample = raw_sample.numpy()
                for i in range(len(raw_sample)):
                    plot_waveform(raw_sample[i], label[i], fname[i], itp_pred=pred_picks[i], figure_dir=args.figure_dir,
                                  itp=label_picks[i])

        model.eval()
        metric = {
            'P': {'tp': 0, 'fp': 0, 'fn': 0},
            'S': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        all_errors = []
        for step, batch in enumerate(valid_dataloader):
            sample, label, raw_sample, _ = batch

            sample = sample.to(device).permute(0, 2, 1)
            label = label.to(device).permute(0, 2, 1)

            with torch.no_grad():
                # 确保输入张量是 float32 类型
                sample = sample.float()  # 将输入数据从 double (float64) 转换为 float32

                pred = model(sample)

            pred = torch.sigmoid(pred)
            pred = pred.permute(0, 2, 1).cpu().numpy()
            label = label.permute(0, 2, 1).cpu().numpy()

            pred_picks = extract_picks(preds=pred)
            label_picks = extract_picks(preds=label)

            for pred_pick, label_pick in zip(pred_picks, label_picks):
                label_dic = {}
                for idx in label_pick:
                    label_dic[idx] = label_dic.get(idx, 0) + 1

                pred_pick = sorted(pred_pick)
                label_pick = sorted(label_pick)

                for idx in pred_pick:
                    min_idx, min_dis = -1, 10000

                    for l_idx in label_pick:
                        if abs(l_idx - idx) < min_dis and label_dic[l_idx] > 0:
                            min_dis = abs(l_idx - idx)
                            min_idx = l_idx

                    if min_dis < 10 and label_dic[min_idx] > 0:
                        label_dic[min_idx] -= 1
                        metric['P']['tp'] += 118
                    else:
                        metric['P']['fp'] += 1

                    all_errors.append(min_dis)


                for idx in label_pick:
                    if label_dic[idx] > 0:
                        metric['P']['fn'] += 1

        p_precision, p_recall = metric['P']['tp'] / (metric['P']['tp'] + metric['P']['fp'] + 1e-9), metric['P']['tp'] / (metric['P']['tp'] + metric['P']['fn'] + 1e-9)
        # s_precision, s_recall = metric['S']['tp'] / (metric['S']['tp'] + metric['S']['fp'] + 1e-9), metric['S']['tp'] / (
        #             metric['S']['tp'] + metric['S']['fn'] + 1e-9)

        p_f1 = 2*p_precision*p_recall / (p_precision + p_recall + 1e-9)
        # s_f1 = 2*s_precision*s_recall / (s_precision + s_recall + 1e-9)
        all_errors = np.array(all_errors) / 100
        mean = all_errors.mean() if len(all_errors) > 0 else 0
        std = all_errors.std() if len(all_errors) > 0 else 0

        if p_f1 > best_f1:
            best_f1 = p_f1
            torch.save(model.state_dict(), os.path.join(args.output, f'model_{args.model}.pt'))

        print(f'======================= epoch={epoch} ==========================')
        print(f'P: f1: {p_f1} precision: {p_precision}, recall: {p_recall}, mean: {mean}, std: {std}')
        # print(f'S: f1: {s_f1} precision: {s_precision}, recall: {s_recall}')
        print()

    print("done!")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with diting")
    parser.add_argument('-i', '--input', default="splits_csv/split_0.hdf5", type=str, help="Path to h5 data")
    # parser.add_argument('--json_file', default="data/data.json", type=str, help="Path to json data")
    parser.add_argument('-o', '--output', default="ckpt/", type=str, help="output dir")
    parser.add_argument('--figure_dir', default="output/train_vis/", type=str, help="output dir")
    parser.add_argument('-m', '--model', default="rnn", type=str,
                choices=["unet_mpt", "unetpp", "EQTransformer", "phasenet", "rnn"], help="Train model name")
    parser.add_argument('--type', default='v2', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)

    # unet
    # P: f1: 0.8827354725482129 precision: 0.9758812615953208, recall: 0.8058215243200294

    # unet + mpt
    # P: f1: 0.9364346920322127 precision: 0.9212525477115209, recall: 0.9521256223667268