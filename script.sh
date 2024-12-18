
# PanNuke + Res50 + Coarse-to-fine regression + Holistic Boundary-aware Branch
# train
bash ./tools/dist_train.sh configs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10.py 4 --launcher pytorch --work_dir work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10

# test
bash tools/dist_test.sh configs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10.py ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10/latest.pth 4 --out work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10/res.pkl --eval segm

# compute_stats
python3 tools/analyze_results.py configs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10.py ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar_heatmap_5_10/res.pkl ./vis
python3 tools/compute_stats.py --pred_dir vis/pred/ --true_dir vis/gt/

# PanNuke + Res50 + Coarse-to-fine regression  
# train
bash ./tools/dist_train.sh configs/panNuke/polar_init_refine_r50_centerness_polar.py 4 --launcher pytorch --work_dir work_dirs/panNuke/polar_init_refine_r50_centerness_polar

# test
bash tools/dist_test.sh configs/panNuke/polar_init_refine_r50_centerness_polar.py ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar/latest.pth 4 --out ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar/res.pkl --eval segm

# compute_stats
python3 tools/analyze_results.py configs/panNuke/polar_init_refine_r50_centerness_polar.py ./work_dirs/panNuke/polar_init_refine_r50_centerness_polar/res.pkl ./vis
python3 tools/compute_stats.py --pred_dir vis/pred/ --true_dir vis/gt/


