
echo "Enter the path to orfd dataset: (ex: E:\\BaiduNetdiskDownload\\ORFD_Dataset_ICRA2022_ZIP)"
read data_root
echo "data_root: $data_root"

echo "Enter the mode: (ex: training/validation)"
read mode
echo "mode: $mode"

# python ./data_prep/orfd/orfd_surface_normal_from_depth.py --data_root $data_root --mode $mode
python ./data_prep/orfd/orfd_foot_print.py --data_root $data_root --mode $mode
#python ./data_prep/orfd/orfd_super_pixel_from_rgb.py --data_root $data_root --mode $mode