# This script invokes the bootsrapped training and testing on the reduced tables
# The output(results) are stored to the results folder that will be printed

train_table_feat=$1 #"feat\$dr_rpca_bert_16_D_20\$T_20\$user_id"
#dr="rpca"
#k="16"
outcomes=$2

if [[ ${3} -eq "1" ]]
then
    task_type="reg"
    model="ridgehighcv"
else
    task_type="classifiers"
    model="lr"
fi;

outcomes_=${outcomes// /$'_'}

models_path_prefix="/data/avirinchipur/ContextualEmbeddingDR3/models"
results_path_prefix="/data/avirinchipur/ContextualEmbeddingDR3/results"

train_table_arr=(${train_table_feat//$/ })
train_table_name=(${train_table_arr[2]})

emb_model_arr=(${train_table_arr[1]//_/ })
emb_model_name=(${emb_model_arr[2]})


if [ ${emb_model_arr[0]} == "dr" ]; then
    dr=emb_model_arr[1]
    pickle_file_name=${train_table_arr[1]:3:-5}
    k=emb_model_arr[3]
else
    echo "Can't recognize the dr. Exitting..."
    exit 0
fi

group_id_field=(${train_table_arr[3]})

task_number_arr=(${train_table_name//_/ })
task_number=(${task_number_arr[1]})

outcome_table_name=(${task_number}"_outcomes")

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

train_command="python3 ~/NLP/dlatk/dlatkInterface.py -d dimRed_contextualEmb -t ${train_table_name} -c ${group_id_field} --group_freq_thresh ${gft} -f '${train_table_feat}' --outcome_table ${outcome_table_name}  --outcomes ${outcomes} --train_${task_type} --model ${model} --train_bootstraps 10 --where 'r10pct_test_fold is NOT NULL' --train_bootstraps_ns 100 200 500 1000 2000 5000 10000 --no_standardize --save_models --picklefile ${models_path_prefix}/${task_name}/bs10.${task_name}.${outcomes_}.${pickle_file_name}_${train_table_name}_nostd.pickle"

test_command="python3 ~/NLP/dlatk/dlatkInterface.py -d dimRed_contextualEmb -t ${train_table_name} -c ${group_id_field} --group_freq_thresh ${gft} -f '${train_table_feat}' --outcome_table ${outcome_table_name}  --outcomes ${outcomes} --predict_${task_type} --where 'facet_fold = 1' --load --picklefile ${models_path_prefix}/${task_name}/bs10.${task_name}.${outcomes_}.${pickle_file_name}_${train_table_name}_nostd.pickle > ${results_path_prefix}/${task_name}/${pickle_file_name}_domain_bootstrap_${outcomes_}_${train_table_name}_nostd.txt"

#echo "----------------------------------------------"

echo "${train_command}"

#echo "----------------------------------------------"

echo "${test_command}"