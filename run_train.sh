#python train.py -m unet_mpt -i data/merge.hdf5 -o ./output --figure_dir ./output/train_vis/ --plot
#DAS-Net
python train.py -m unet_mpt -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/
#D-Net
python train.py -m unet -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/
#U-Net
python train.py -m unet -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/ --type v1
#DA-Net
python train.py -m unet_mpt -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/
#DAMX
python train.py -m unet_mpt -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/ --type v1
#EQT    111
python train.py -m EQTransformer -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/
#phasenet
python train.py -m phasenet -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/

python train_pre.py -m unet -i data/data_test.hdf5 -o ./output --figure_dir ./output/train_vis/

python train_pre.py -m unet -i data/non_naturaldata.hdf5 -o ./output --figure_dir ./output/train_vis/
python train_pre.py -m unet -i splits_csv/split_0.hdf5 -o ./output --figure_dir ./output/train_vis/

python train.py -m NestedUNet -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/

python train.py -m phasenet -i data/chunk2.hdf5 -o ./output --figure_dir ./output/train_vis/



python train_instance.py -m unet -i splits_csv/split_0.hdf5 -o ./output --figure_dir ./output/train_vis/ --type v1
python train_instance.py -m unet_mpt -i splits_csv/split_0.hdf5 -o ./output --figure_dir ./output/train_vis/ --type v1
python train_instance.py -m EQTransformer -i splits_csv/split_0.hdf5 -o ./output --figure_dir ./output/train_vis/ --type v1

python train_instance.py -m rnn -i splits_csv/split_0.hdf5 -o ./output --figure_dir ./output/train_vis/ --type v1

python train_instance.py -m unet_mpt -i data/processed_data.h5 -o ./output --figure_dir ./output/train_vis/


python train_instance.py -m unet_mpt -i splits_csv/split_0.hdf5 -o ./output --figure_dir ./output/train_vi/ --type v1

python train_pre.py -m unet_mpt -i data/processed_data.h5 -o ./output --figure_dir ./output/train_vis_pre/

