cd /apdcephfs/private_amandaaluo/PolarMask

python3 setup.py develop

# sh ./tools/dist_train.sh configs/polarmask/4gpu/polar_768_1x_r50.py 4 --launcher pytorch --work_dir /apdcephfs/private_amandaaluo/polarmask/work_dirs/Oct302/polar_768_1x_r50_4gpu

# sh tools/dist_test.sh configs/polarmask_refine/1gpu/polarmask_refine_l2_normalize_init3_refine6.py work_dirs/polarmask_refine_l2_normalize_init3_refine6/latest.pth 1 --out work_dirs/polarmask_refine_l2_normalize_init3_refine6/res_refine_center.pkl --eval segm

# sh tools/dist_test.sh configs/polarmask_refine/1gpu/polarmask_refine_l2_normalize_init3_refine6_mask_init.py work_dirs/polarmask_refine_l2_normalize_init3_refine6/latest.pth 1 --out work_dirs/polarmask_refine_l2_normalize_init3_refine6/res_center.pkl --eval segm

sh tools/dist_test.sh configs/polarmask_refine/1gpu/polarmask_refine_l2_normalize_3layer_heatmap_contour.py work_dirs/polarmask_refine_l2_normalize_3layer_heatmap_contour/latest.pth 1 --out work_dirs/polarmask_refine_l2_normalize_3layer_heatmap_contour/res.pkl --eval segm
