#python train.py -m unet -i test_data/merge.hdf5 -o ./output
python test.py -m unet_mpt -i data/chunk3.hdf5 -o ./output --figure_dir ./output/test_vis/
python test.py -m phasenet -i data/chunk3.hdf5 -o ./output --figure_dir ./output/test_vis/
python test.py -m EQTransformer -i data/chunk3.hdf5 -o ./output --figure_dir ./output/test_vis/

python test.py -m unet_mpt -i data/chunk2.hdf5 -o ./output --figure_dir ./output/test_vis/


python test_instance.py -m unet -i splits_csv/split_1.hdf5 -o ./output --figure_dir ./output/test_vis/
python test_instance.py -m EQTransformer -i splits_csv/split_2.hdf5 -o ./output --figure_dir ./output/test_vis/

python test_instance.py -m rnn -i splits_csv/split_2.hdf5 -o ./output --figure_dir ./output/test_vis/



python test.py -m unet_mpt -i data/data_test_csv.hdf5 -o ./output --figure_dir ./output/test_vis/
python test_pre.py -m trans/unet_mpt -i data/processed_data.h5 -o ./output --figure_dir ./output/test_vis/
 ##########################processed_data 是非天然地震数据#############################

 python test.py -m unet_mpt -i data/processed_data.h5 -o ./output --figure_dir ./output/test_vis/
 python test_instance.py -m unet_mpt -i splits_csv/split_1.hdf5 -o ./output --figure_dir ./output/test_vis/
 python test_instance.py -m unet_mpt -i data/processed_data.h5 -o ./output --figure_dir ./output/test_vis/

########################################12.16
python test.py -m unetpp -i data/chunk3.hdf5 -o ./output --figure_dir ./output/test_vis/
 python test.py -m unet_mpt -i data/chunk3.hdf5 -o ./output --figure_dir ./output/test_vis/
 python test.py -m unet_mpt -i data/processed_data.h5 -o ./output --figure_dir ./output/test_vis/




 python test_instance.py -m unet_mpt -i splits_csv/split_1.hdf5 -o ./output --figure_dir ./output/test_vi/
 python test_pre.py -m unet_mpt -i data/processed_data.h5 -o ./output --figure_dir ./output/test_vi/
