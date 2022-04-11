
MY_CMD="python main.py --batch_size 512 --epochs 100 --imbalance --imbalance_memory --fine_label_memory --mode knn_test_fine_label --load_model --load_model_path 48690035_1_128_0.5_200_512_1000_best_test_acc_model --local 2 --no_save"
# /mnt/home/renjie3/Documents/course/cse891/imbalance/LDAM-DRW/results/48690035_1_128_0.5_200_512_1000_statistics.csv

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD
# done