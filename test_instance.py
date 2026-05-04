
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import torch
from models.EQT_instance import EQTransformer
import matplotlib.pyplot as plt
from models.BRNN_instance import BRNN

plt.rc('font', family='serif', serif='Times New Roman')

from models.UNetAS_instance import UNet
# from models.UNETitself import UNet_mpt,UNet
from models.UNetDA import UNet
from models.phasenet_instance import PhaseNet
from models.unetpp import UNetpp
# from models.UNetAS_instance import UNet
from models.DAMS import UNet_mpt
# from models.UNet_v2 import UNetV2, UNet_mptV2
from mydataset.seismic_dataset_instance import SeismicDataset
from mydataset.seismic_dataset_v2 import SeismicDatasetV2
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import random
from datetime import datetime, timedelta
import os 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150


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


def plot_waveform(data, pred, fname, label=None,
                  itp=None, its=None, itps=None,
                  itp_pred=None, its_pred=None, itps_pred=None,
                  figure_dir="./", dt=0.01):
    t = np.arange(0, pred.shape[0]) * dt
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.05, 0.77]

    plt.figure()

    # plt.subplot(411)
    # plt.plot(t, data[:, 0], 'k', label='E', linewidth=0.5)
    # plt.autoscale(enable=True, axis='x', tight=True)
    # tmp_min = np.min(data[:, 0])
    # tmp_max = np.max(data[:, 0])
    # if (itp is not None):
    #     for j in range(len(itp)):
    #         lb = "P" if j == 0 else ""
    #         plt.plot([itp[j] * dt, itp[j] * dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
    # if (itps is not None):
    #     for j in range(len(itps)):
    #         lb = "PS" if j == 0 else ""
    #         plt.plot([itps[j] * dt, its[j] * dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
    # plt.ylabel('Amplitude')
    # plt.legend(loc='upper right', fontsize='small')
    # plt.gca().set_xticklabels([])
    # plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='center',
    #          transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    #
    # plt.subplot(412)
    # plt.plot(t, data[:, 1], 'k', label='N', linewidth=0.5)
    # plt.autoscale(enable=True, axis='x', tight=True)
    # tmp_min = np.min(data[:, 1])
    # tmp_max = np.max(data[:, 1])
    # if (itp is not None):
    #     for j in range(len(itp)):
    #         lb = "P" if j == 0 else ""
    #         plt.plot([itp[j] * dt, itp[j] * dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
    #     # for j in range(len(its)):
    #     #     lb = "S" if j == 0 else ""
    #     #     plt.plot([its[j] * dt, its[j] * dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
    # if (itps is not None):
    #     for j in range(len(itps)):
    #         lb = "PS" if j == 0 else ""
    #         plt.plot([itps[j] * dt, itps[j] * dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
    # plt.ylabel('Amplitude')
    # plt.legend(loc='upper right', fontsize='small')
    # plt.gca().set_xticklabels([])
    # plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='center',
    #          transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

    plt.subplot(413)
    plt.plot(t, data[:, 2], 'k', label='Z', linewidth=0.5)
    plt.autoscale(enable=True, axis='x', tight=True)
    tmp_min = np.min(data[:, 2])
    tmp_max = np.max(data[:, 2])
    if (itp is not None):
        for j in range(len(itp)):
            lb = "P" if j == 0 else ""
            plt.plot([itp[j] * dt, itp[j] * dt], [tmp_min, tmp_max], 'C0', label=lb, linewidth=0.5)
        # for j in range(len(its)):
        #     lb = "S" if j == 0 else ""
        #     plt.plot([its[j] * dt, its[j] * dt], [tmp_min, tmp_max], 'C1', label=lb, linewidth=0.5)
    if (itps is not None):
        for j in range(len(itps)):
            lb = "PS" if j == 0 else ""
            plt.plot([itps[j] * dt, itps[j] * dt], [tmp_min, tmp_max], 'C2', label=lb, linewidth=0.5)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_xticklabels([])
    plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='center',
             transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)

    plt.subplot(414)
    if label is not None:
        plt.plot(t, label[:, 0, 1], 'C0', label='P', linewidth=1)
        plt.plot(t, label[:, 0, 2], 'C1', label='S', linewidth=1)
        if label.shape[-1] == 4:
            plt.plot(t, label[:, 0, 3], 'C2', label='PS', linewidth=1)

    plt.plot(t, pred, '--C0', label='$\hat{P}$', linewidth=1)
    plt.autoscale(enable=True, axis='x', tight=True)
    if (itp_pred is not None):
        for j in range(len(itp_pred)):
            plt.plot([itp_pred[j] * dt, itp_pred[j] * dt], [-0.1, 1.1], '--C0', linewidth=1)
        # for j in range(len(its_pred)):
        #     plt.plot([its_pred[j] * dt, its_pred[j] * dt], [-0.1, 1.1], '--C1', linewidth=1)
    if (itps_pred is not None):
        for j in range(len(itps_pred)):
            plt.plot([itps_pred[j] * dt, itps_pred[j] * dt], [-0.1, 1.1], '--C2', linewidth=1)
    plt.ylim([-0.05, 1.05])
    plt.text(text_loc[0], text_loc[1], '(iv)', horizontalalignment='center',
             transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.gcf().align_labels()

    try:
        plt.savefig(os.path.join(figure_dir, fname + '.png'), bbox_inches='tight')
        plt.savefig(os.path.join(figure_dir, fname + '.pdf'), bbox_inches='tight')

    except FileNotFoundError:
        os.makedirs(os.path.dirname(os.path.join(figure_dir, fname)), exist_ok=True)
        plt.savefig(os.path.join(figure_dir, fname + '.png'), bbox_inches='tight')
        plt.savefig(os.path.join(figure_dir, fname + '.pdf'), bbox_inches='tight')


    plt.close()
    return 0


def main(args):
    fix_random_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    csv_file_path = "D:/X/p_wave/splits_csv/split_0.csv"

    if args.type == 'v1':
        valid_dataset = SeismicDataset(args.input,csv_file=csv_file_path, start=0, end=500, phase='test')
    else:
        valid_dataset = SeismicDatasetV2(args.input, args.json_file, start=0, end=9000, phase='test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=8)

    print(f'Val data size: {len(valid_dataset)}')

    if args.model == 'unet':
        model = UNet() if args.type == 'v1' else UNet()
    elif args.model == 'unet_mpt':
        model = UNet_mpt() if args.type == 'v1' else UNet_mpt()
    elif args.model == 'EQTransformer':
        model =EQTransformer() if args.type == 'v1' else EQTransformer()
    elif args.model == 'phasenet':
        model =PhaseNet() if args.type == 'v1' else PhaseNet()
    elif args.model == 'rnn':
        model =BRNN() if args.type == 'v1' else BRNN()
    elif args.model == 'unetpp':
        model =UNetpp() if args.type == 'v1' else UNetpp()
    else:
        raise ValueError

    state_dict = torch.load(os.path.join(args.output, f'model_{args.model}.pt'))
    msg = model.load_state_dict(state_dict, strict=False)
    print('load ckpt: ', msg)

    model.to(device)

    model.eval()
    metric = {
        'P': {'tp': 0, 'fp': 0, 'fn': 0},
        'S': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    pred_list, label_list = [], []
    all_errors = []
    all_names = []
    for step, batch in enumerate(valid_dataloader):
        sample, label, raw_sample, fname = batch

        sample = sample.to(device).permute(0, 2, 1)
        label = label.to(device).permute(0, 2, 1)
        sample = sample.to(torch.float32)

        with torch.no_grad():
            pred = model(sample)

        pred = torch.sigmoid(pred)
        pred = pred.permute(0, 2, 1).cpu().numpy()
        label = label.permute(0, 2, 1).cpu().numpy()

        pred_picks = extract_picks(preds=pred)
        label_picks = extract_picks(preds=label)

        for pred_pick, label_pick, name in zip(pred_picks, label_picks, fname):
            label_dic = {}
            for idx in label_pick:
                label_dic[idx] = label_dic.get(idx, 0) + 1

            pred_pick = sorted(pred_pick)
            label_pick = sorted(label_pick)

            for idx in pred_pick:
                min_idx, min_dis = -1, 10000
                error = np.nan

                for l_idx in label_pick:
                    if abs(l_idx - idx) < min_dis and label_dic[l_idx] > 0:
                        min_dis = abs(l_idx - idx)
                        min_idx = l_idx
                        error = l_idx - idx

                if min_dis < 10 and label_dic[min_idx] > 0:
                    label_dic[min_idx] -= 1
                    metric['P']['tp'] += 1
                elif min_idx > 0:
                    label_dic[min_idx] -= 1
                    metric['P']['fp'] += 1
                    metric['P']['fn'] += 1
                if not np.isnan(error):
                    all_errors.append(error)
                all_names.append(name)
                pred_list.append(idx)
                label_list.append(min_idx)
            # for idx in range(len(label_dic)):
            #     # if label_dic[idx] == 0 and pred_list[idx] == -1:  # 假设 -1 代表负类预测
            #     #     metric['N']['tn'] += 1
            #     print(f"Current index (idx): {idx}")
            #     print(f"Length of pred_list: {len(pred_list)}")
            #     print(f"Label dictionary: {label_dic}")
            #     if label_dic.get(idx, 0) == 0 and pred_list[idx] == -1:
            #         metric['P']['tn'] += 1

            # for idx in label_pick:
            #     if label_dic[idx] > 0:
            #         metric['P']['fn'] += 1
            #         pred_list.append(-1)
            #         label_list.append(idx)
            #         all_names.append(name)

        # save visualization results
        raw_sample = raw_sample.numpy()
        for i in range(len(raw_sample)):
            plot_waveform(raw_sample[i], label[i], fname[i], itp_pred=pred_picks[i], figure_dir=args.figure_dir)


    p_precision, p_recall = metric['P']['tp'] / (metric['P']['tp'] + metric['P']['fp'] + 1e-9), metric['P']['tp'] / (metric['P']['tp'] + metric['P']['fn'] + 0.1)
    # s_precision, s_recall = metric['S']['tp'] / (metric['S']['tp'] + metric['S']['fp'] + 1e-9), metric['S']['tp'] / (
    #             metric['S']['tp'] + metric['S']['fn'] + 1e-9)

    TP = metric['P']['tp']
    # TN = metric['P']['tn']  # 假设你有负类的真阴性
    FP = metric['P']['fp']
    FN = metric['P']['fn']

    # accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-9)  # 加上1e-9以避免除以零

    p_f1 = 2 * p_precision * p_recall / (p_precision + p_recall + 1e-9)
    # s_f1 = 2*s_precision*s_recall / (s_precision + s_recall + 1e-9)
    print(all_errors)
    all_errors = np.array(all_errors) / 100
    mean = all_errors.mean() if len(all_errors) > 0 else 0
    std = all_errors.std() if len(all_errors) > 0 else 0
    print(f'P: f1: {p_f1} precision: {p_precision}, recall: {p_recall}')
    # print(f'S: f1: {s_f1} precision: {s_precision}, recall: {s_recall}')
    print()

    res_df = pd.DataFrame()
    res_df['名称'] = all_names
    res_df['预测值'] = pred_list
    res_df['真实值'] = label_list

    res_df.to_csv('./output/预测结果.csv', index=False)

    print("done!")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with diting")          
    parser.add_argument('-i', '--input', default="data/diting.h5", type=str, help="Path to h5 data")
    parser.add_argument('--json_file', default="data/first.json", type=str, help="Path to json data")
    parser.add_argument('-o', '--output', default="output/", type=str, help="output dir")
    parser.add_argument('--figure_dir', default="output/test_vis/", type=str, help="output dir")
    parser.add_argument('-m', '--model', default="rnn", type=str,
                choices=["unet_mpt", "unet","EQTransformer", "phasenet", "rnn","unetpp"], help="Train model name")
    parser.add_argument('--plot', default=True, type=bool, help="Whether plot training figure")
    parser.add_argument('--type', default='v1', type=str)
    args = parser.parse_args()      
    main(args)