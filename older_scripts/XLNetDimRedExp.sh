#author: @adithya8

if [ "$1" -eq "18" ]; 
then
    declare -A db=(["d"]="clp18_adi" ["t"]="tr_a11essays" ["c"]="clp18_id")
    declare -A dbTables=(["tr_a11"]="tr_a11essays")
    declare -A lexTables=(["tr_a11"]="tr_a11_xln_")
    dimRedModel=$2
    msgk=$3
else
    declare -A db=(["d"]="clp19_adi" ["t"]="task_A" ["c"]="user_id") 
    declare -A dbTables=(["A"]="task_A" ["C"]="task_Cfil" ["At"]="task_A_title" ["Ct"]="task_Cfil_title")
    declare -A lexTables=(["A"]="task_A_xln_" ["C"]="task_Cfil_xln_" ["At"]="taskAt_xln_" ["Ct"]="taskCt_xln_")
    dimRedModel=$2
    msgk=$3
    titlek=$4         
fi

declare -A dimRedModels=(["fa"]="fa" ["pca"]="pca" ["nmf"]="nmf")

resultFile="XLN_${dimRedModel}_${msgk}_${titlek}.txt"

echo "$resultFile"

echo "$1_$2_$3"

if test -d "~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/"; then 
    echo "Directory Exists"
else        
    eval "mkdir ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/"
    if test -d "~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/"; then 
        echo "Directory Created"
    fi
fi 

for i in ${!dbTables[@]}
 do
    dbTable=${dbTables[$i]}
    lexTable=${lexTables[$i]}
    
    if [[ $dbTable =~ "title" ]]
    then
        k=${titlek}
    else
        k=${msgk}
    fi

    lexTableCr="~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${dbTable} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$xlnet_ba_ca_memamiL10co\$${dbTable}\$${db["c"]}\$16to16' --fit_reducer --model ${dimRedModel} --n_components ${k} --reducer_to_lexicon ${lexTable}${dimRedModel}${k}"
    echo "${lexTableCr}"
    eval "${lexTableCr}"
    echo "----------------------------------------------------------------------"
    weightLex="~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${dbTable} -c ${db["c"]} --group_freq_thresh 0 --word_table 'feat\$xlnet_ba_ca_memamiL10co\$${dbTable}\$${db["c"]}\$16to16' --add_lex -l ${lexTable}${dimRedModel}${k} --weighted_lex"
    echo "${weightLex}"
    eval "${weightLex}"
    echo "----------------------------------------------------------------------"
done

if [ "$1" -eq "18" ];
then
    finalCommand="~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["tr_a11"]}${dimRedModel}${msgk}_w\$${db["t"]}\$${db["c"]}\$xlne' --outcome_table tr_variables --outcomes a11_bsag_total --combo_test_reg --model ridgehighcv --folds 10 > ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/${resultFile}"
    #saveModelCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["tr_a11"]}${dimRedModel}${msgk}_w\$${db["t"]}\$${db["c"]}\$xlne'  --outcome_table task_labels_full --outcomes label --train_classifiers --model ridge --save_model --picklefile ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/xln_${dimRedModel}_${msgk}_${titlek}.pickle"
else 
    finalCommand="~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["A"]}${dimRedModel}${msgk}_w\$${dbTables["A"]}\$${db["c"]}\$xlne' 'feat\$cat_${lexTables["At"]}${dimRedModel}${titlek}_w\$${dbTables["At"]}\$${db["c"]}\$xlne' 'feat\$cat_${lexTables["Ct"]}${dimRedModel}${titlek}_w\$${dbTables["Ct"]}\$${db["c"]}\$xlne' 'feat\$cat_${lexTables["C"]}${dimRedModel}${msgk}_w\$${dbTables["C"]}\$${db["c"]}\$xlne' --outcome_table task_labels_full --outcomes label --nfold_classifiers --model lr  --folds 10 > ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/${resultFile}"    
    #saveModelCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_task_A_xln_${dimRedModel}${msgk}_w\$task_A\$user_id\$xlne' 'feat\$cat_taskAt_xln_${dimRedModel}${titlek}_w\$task_A_title\$user_id\$xlne' 'feat\$cat_taskCt_xln_${dimRedModel}${titlek}_w\$task_Cfil_title\$user_id\$xlne' 'feat\$cat_task_Cfil_xln_${dimRedModel}${msgk}_w\$task_Cfil\$user_id\$xlne' --outcome_table task_labels_full --outcomes label --train_classifiers --model lr --save_model --picklefile ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/xln_${dimRedModel}_${msgk}_${titlek}.pickle"         
fi
echo "${finalCommand}" 
eval "${finalCommand}" 
echo "----------------------------------------------------------------------"

if [ "$1" -eq "18" ];
then
    eval "cat ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/${resultFile} | grep \'R\':"
else    
    eval "cat ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/${resultFile} | grep \'f1\':"
fi