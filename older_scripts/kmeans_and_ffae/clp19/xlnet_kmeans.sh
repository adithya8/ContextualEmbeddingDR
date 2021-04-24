echo "Running kmeans on task_A table"
python3 /users2/aravula/clp19_results/kmeans.py "feat\$xlnet_ba_ca_memamiL10co\$task_A\$user_id\$16to16.csv_dense.csv" "feat\$xlnet_ba_ca_memamiL10co\$task_A_test\$user_id\$16to16.csv_dense.csv" $1
echo "Running kmeans on task_Cfil table"
python3 /users2/aravula/clp19_results/kmeans.py "feat\$xlnet_ba_ca_memamiL10co\$task_Cfil\$user_id\$16to16.csv_dense.csv" "feat\$xlnet_ba_ca_memamiL10co\$task_Cfil_test\$user_id\$16to16.csv_dense.csv" $1
echo "Running kmeans on task_A_title table"
python3 /users2/aravula/clp19_results/kmeans.py "feat\$xlnet_ba_ca_memamiL10co\$task_A_title\$user_id\$16to16.csv_dense.csv" "feat\$xlnet_ba_ca_memamiL10co\$task_A_title_test\$user_id\$16to16.csv_dense.csv" $2
echo "Running kmeans on task_Cfil_title table"
python3 /users2/aravula/clp19_results/kmeans.py "feat\$xlnet_ba_ca_memamiL10co\$task_Cfil_title\$user_id\$16to16.csv_dense.csv" "feat\$xlnet_ba_ca_memamiL10co\$task_Cfil_title_test\$user_id\$16to16.csv_dense.csv" $2
