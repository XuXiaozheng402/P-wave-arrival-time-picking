import time
import numpy as np 
import matplotlib.pyplot as plt
import torch
import re
#############################instance#####################
# from models.UNetAS_instance import UNet, UNet_mpt
from models.UNetAS import UNet,UNet_mpt
from models.EQT_instance import EQTransformer
from models.phasenet import PhaseNet
from models.UNet_v2 import UNetV2, UNet_mptV2
# from mydataset.seismic_dataset import SeismicDataset
from mydataset.seismic_dataset_kuang import SeismicDataset_kuang

# from mydataset.seismic_dataset_v2 import SeismicDatasetV2
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import random
from test import plot_waveform
import os 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
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
    dx = x[1:] - x[:-1]
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
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1] 
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0 
        ind = np.sort(ind[~idel])

    return ind, x[ind]

def extract_picks(preds):
    mph = {"P": 0.3, "S": 0.3, "PS": 0.3}
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

#####################一共15373数据##########################
    if args.type == 'v1':
        # train_dataset = SeismicDataset(args.input, start=20, end=80, phase='train')
        # valid_dataset = SeismicDataset(args.input, start=801, end=900, phase='test')
        train_dataset = SeismicDataset_kuang(args.input, start=0, end=7000, phase='train')
        valid_dataset = SeismicDataset_kuang(args.input, start=7001, end=10373, phase='test')


    else:
        train_dataset = SeismicDataset_kuang(args.input, start=0, end=11000, phase='train')
        valid_dataset = SeismicDataset_kuang(args.input, start=11001, end=15373, phase='test')
        # 检查数据集的长度
    print(f"数据集长度: {len(train_dataset)}")


    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=8)

    print(f'Train data size: {len(train_dataset)}, val data size: {len(valid_dataset)}')

    lossfn = torch.nn.BCEWithLogitsLoss()

    if args.model == 'unet':
        model = UNet() if args.type == 'v1' else UNetV2()
    elif args.model == 'unet_mpt':
        model = UNet_mpt() if args.type == 'v1' else UNet_mptV2()
    elif args.model == 'EQTransformer':
        model = EQTransformer() if args.type == 'v1' else EQTransformer()
    elif args.model == 'phasenet':
        model = PhaseNet() if args.type == 'v1' else PhaseNet()
    else:
        raise ValueError("Invalid model choice")

    # 加载预训练权重
    pretrained_weights_path = 'D:/X/p_wave/output/model_unet_mptas.pt'  # 替换为实际的预训练模型路径
    # pretrained_weights_path = 'D:/X/p_wave/output/UNet/model_unet.pt'  # 替换为实际的预训练模型路径
    ##################################instance_right######################################################################################

    # pretrained_weights_path = 'D:/X/p_wave/output/instance/unet_mpt/model_unet_mpt.pt'  # 预训练模型路径

    model.load_state_dict(torch.load(pretrained_weights_path))
    # model = torch.load(pretrained_weights_path)  # 直接加载整个模型
    def freeze_srm_layer(model):
        for name, param in model.named_parameters():
            if "cfc" in name or "bn.weight" in name or "bn.bias" in name:
                param.requires_grad = False  # 冻结参数
            else:
                param.requires_grad = True  # 其他层的参数可以训练
    freeze_srm_layer(model)
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    ########################微调###################################
    import re


    freeze_layers = ['down_1', 'down_2', 'down_3', 'down_4','up_1','up_2','up_3']
    for name, param in model.named_parameters():
        if any(re.search(layer, name) for layer in freeze_layers):
            param.requires_grad = False
        else:
            param.requires_grad = True  # 这些层的参数被解冻

###################冻结AMS###########################################################
    import re

    # 冻结 SRMLayer
    # for name, param in model.named_parameters():
    #     if "SRMLayer" in name:  # 关键部分：确保 SRMLayer 被正确匹配
    #         param.requires_grad = False  # 冻结参数
    #     else:
    #         param.requires_grad = True  # 其他层的参数可以训练

    # 仅更新 requires_grad=True 的参数
    # for name, param in model.named_parameters():
    #     if "layers.4.bn" in name:  # 匹配 SRMLayer 中的 BatchNorm1d 参数
    #         param.requires_grad = False  # 冻结参数
    #     else:
    #         param.requires_grad = True  # 其他层的参数可以训练
    #
    # # 打印模型参数，检查冻结是否生效
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    # optim = torch.optim.Adam((param for name, param in model.named_parameters() if param.requires_grad), lr=1e-5,
    #                          weight_decay=1e-3)
    ##############################################################################
    model.eval()  # 让 BN 层保持固定
    # print(model)
    ########################微调#################################
    #######################全调######################################
    # （可选）冻结某些层
    # for param in model.parameters():
    #     param.requires_grad = True
    # optim = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-3)


    #####################全调####################################


##############################################直接用#####################
    # （可选）冻结某些层
    # for param in model.parameters():
    #     param.requires_grad = True
    optim = torch.optim.Adam((param for name, param in model.named_parameters() if param.requires_grad), lr=1e-4,
                             weight_decay=1e-3)
    #########################################################



    model.to(device)
    lossfn.to(device)
    best_f1 = 0





    ##################模型训练#####################################

    for epoch in range(50):
        model.train()
        for step, batch in enumerate(train_dataloader):
            sample, label, raw_sample, fname = batch

            sample = sample.to(device).permute(0, 2, 1)
            label = label.to(device).permute(0, 2, 1)

            pred = model(sample)
            loss = lossfn(pred, label)
            loss.backward()

            optim.step()
            optim.zero_grad()

            if step % 300 == 0:
                print('loss: ', loss.item())

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

                    if min_idx == -1:
                        metric['P']['fp'] += 1
                    elif min_dis <= 20:
                        metric['P']['tp'] += 1
                        label_dic[min_idx] -= 1
                        all_errors.append(min_dis)
                    else:
                        metric['P']['fp'] += 1
                        metric['P']['fn'] += 1

        # precision = metric['P']['tp'] / (metric['P']['tp'] + metric['P']['fp'])
        if metric['P']['tp'] + metric['P']['fp'] > 0:
            precision = metric['P']['tp'] / (metric['P']['tp'] + metric['P']['fp'])
        else:
            precision = 0  # 或者可以选择其他方式处理，比如设置为 None 或忽略这个数据点
        if metric['P']['tp'] + metric['P']['fp'] > 0:
            recall = metric['P']['tp'] / (metric['P']['tp'] + metric['P']['fn'])
        else:
            recall = 0  # 或者可以选择其他方式处理，比如设置为 None 或忽略这个数据点
        if precision + recall > 0:

            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        print(f'======================= epoch={epoch} ==========================')
        print(f'P: f1: {f1} precision: {precision}, recall: {recall}')
        # print(f'S: f1: {s_f1} precision: {s_precision}, recall: {s_recall}')
        print()


        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pt'))

            # torch.save(model, os.path.join(args.output, 'best_model.pt'))

    print(f"Best F1 score = {best_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Seismic Event Detection Model")
    parser = argparse.ArgumentParser(description="Train with diting")
    parser.add_argument('-i', '--input', default="data/processed_data.h5", type=str, help="Path to h5 data")
    # parser.add_argument('--json_file', default="data/data.json", type=str, help="Path to json data")
    parser.add_argument('-o', '--output', default="ckpt/", type=str, help="output dir")
    parser.add_argument('--figure_dir', default="output/train_vis/", type=str, help="output dir")
    parser.add_argument('-m', '--model', default="rnn", type=str,
                choices=["unet_mpt", "unet","EQTransformer", "phasenet"], help="Train model name")
    parser.add_argument('--type', default='v1', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)
