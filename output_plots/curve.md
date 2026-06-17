# 跑训练集上的三方法比对曲线

python compare_curves.py \
  --split_log OpenPCDet/output/kitti_models/pv_rcnn_train_as_test/default/eval/epoch_no_number/train/default/log_eval_split_20260601-082007.txt \
  --split_csv GPCC/estimate_results/estimate_average_results.csv \
  --jucp_txt_dir OpenPCDet/output/kitti_models/pv_rcnn_train_as_test/default/eval/epoch_no_number/train/default \
  --jucp_csv_dir OpenPCDet/tools \
  --out output_plots/compare_2method_train.png