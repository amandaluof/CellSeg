cd /apdcephfs/private_amandaaluo/PolarMask

python3 setup.py develop

pip install future tensorboard

sh ./tools/dist_train.sh configs/polarmask_refine/1gpu/polar_init_refine_r101_centerness_polar_heatmap_revise_aug.py  1 --launcher pytorch --work_dir ./work_dirs/debug

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_polar_both.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_polar_both

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_init_l2_normalize_polar_gt.py  4 --launcher pytorch --work_dir ./work_dirs/polarmask_init_l2_normalize_polar_gt

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_3layer_heatmap_contour_upsample.py  4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_3layer_heatmap_contour_upsample

# sh tools/dist_train.sh configs/polarmask/4gpu/polar_768_1x_r50.py 4 --launcher pytorch --work_dir /apdcephfs/private_amandaaluo/PolarMask/work_dirs/polar_768_1x_r50/toy

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_3layer_refine_only_refine_loss.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_3layer_refine_only_refine_loss

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_init5_refine5.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_init5_refine5

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_3layer_heatmap_contour.py  4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_3layer_heatmap_contour

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_polar_xy_baseline.py  4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_polar_xy_baseline

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_init3_refine6_polar_xy.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_init3_refine6_polar_xy

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_init3_refine6_extreme.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_init3_refine6_extreme

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_init3_refine6_gradient8.py  4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_init3_refine6_gradient8

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_P3_all_instances_contour_heatmap.py  4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_P3_all_instances_contour_heatmap

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_init3_refine6_last_feat.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_init3_refine6_last_feat.py

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_init3_refine6_nonseperate.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_init3_refine6_nonseperate

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_3layer_refine.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_3layer_refine

#sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l2_normalize_init3_refine6.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l2_normalize_init3_refine6

# sh ./tools/dist_train.sh configs/polarmask_refine/4gpu/polarmask_refine_l1_normalize_beta10_init3_refine6.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_refine_l1_normalize_beta10_init3_refine6

# sh ./tools/dist_train.sh configs/polarmask_gt_cartesian/4gpu/polarmask_gt_contour_cartesian_l1_beta10.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_gt_contour_cartesian_l1_beta10

# sh ./tools/dist_train.sh configs/polarmask_gt_cartesian/4gpu/polarmask_gt_contour_cartesian_l1_normalize8_beta8.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_gt_contour_cartesian_l1_normalize8_beta8

# sh ./tools/dist_train.sh configs/polarmask_gt_cartesian/4gpu/polarmask_gt_contour_cartesian_l1_normalize8_beta10.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_gt_contour_cartesian_l1_normalize8_beta10

# sh ./tools/dist_train.sh configs/polarmask_gt_cartesian/4gpu/polarmask_gt_contour_cartesian_l2_normalize8_explore0_inner0.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_gt_contour_cartesian_l2_normalize8_explore0_inner0

# sh ./tools/dist_train.sh configs/polarmask_gt_cartesian/4gpu/polarmask_gt_contour_cartesian_l2_normalize10_explore0_inner0.py 4 --launcher pytorch --work_dir ./work_dirs/polarmask_gt_contour_cartesian_l2_normalize10_explore0_inner0

# sh ./tools/dist_train.sh configs/polarmask_gt_cartesian/4gpu/polarmask_gt_contour_cartesian_l2_explore0_inner0.py  4 --launcher pytorch --work_dir ./work_dirs/polarmask_gt_contour_cartesian_l2_explore0_inner0

# sh tools/dist_train.sh configs/polarmask/4gpu/polar_768_1x_r50.py 4 --launcher pytorch --work_dir /apdcephfs/private_amandaaluo/PolarMask/work_dirs/polar_768_1x_r50
