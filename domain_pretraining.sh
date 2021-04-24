# This script trains the specified reduction algorithm on the domain table provided and applies it on the corresponding Train (combined with the test table) that's provided
# The applies table is stored in the same mysql DB.
# TODO: Add arg for giving the feat_table_name

#DEBUGGING INPUTS
#domain_table_feat="feat\$bert_ba_un_meL11con\$D_20\$user_id\$16to16"
#train_table_feat="feat\$bert_ba_un_meL11con\$T_20\$user_id\$16to16"
#dr=pca
#k=16

domain_table_feat=$1
train_table_feat=$2
dr=$3
k=$4

models_path_prefix="/data/avirinchipur/ContextualEmbeddingDR3/models"

domain_table_arr=(${domain_table_feat//$/ })
domain_table_name=(${domain_table_arr[2]})

train_table_arr=(${train_table_feat//$/ })
train_table_name=(${train_table_arr[2]})

task_number_arr=(${train_table_arr[2]//_/ })
task_number=(${task_number_arr[1]})

if [[ task_number -eq "20" ]]
then 
    task_name="fb20"
    gft="1000"
elif [[ task_number -eq "19" ]]
then
    task_name="clp19"
    gft="0"
elif [[ task_number -eq "18" ]]
then
    task_name="clp18"
    gft="0"
fi;

emb_model_arr=(${train_table_arr[1]//_/ })
emb_model_name=(${emb_model_arr[0]})

group_id_field=(${train_table_arr[3]})

feat_table_name=(${emb_model_name}_${k}_${domain_table_name})

learn_dr_command="python3 ~/NLP/dlatk/dlatkInterface.py -d dimRed_contextualEmb -t ${domain_table_name} -c ${group_id_field} --group_freq_thresh ${gft} -f '${domain_table_feat}' --model ${dr} --n_components ${k} --fit_reducer --save_models --picklefile ${models_path_prefix}/${task_name}/${dr}_${feat_table_name}.pickle"

apply_dr_command="python3 ~/NLP/dlatk/dlatkInterface.py -d dimRed_contextualEmb -t ${train_table_name} -c ${group_id_field} --group_freq_thresh ${gft} -f '${train_table_feat}' --transform_to_feats ${feat_table_name} --load --picklefile ${models_path_prefix}/${task_name}/${dr}_${feat_table_name}.pickle"

#echo "----------------------------------------------"

echo "${learn_dr_command}"

#echo "----------------------------------------------"

echo "${apply_dr_command}"

#dr_train_table="feat\$dr_${dr}_${feat_table_name}\$${group_id_field}"
#echo "${dr_train_table}"