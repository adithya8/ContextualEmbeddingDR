# author: @adithya8

if [ "$1" -eq "18" ]; 
then
    declare -A db=(["d"]="clp18_adi" ["t"]="tr_a11essays" ["c"]="clp18_id" ["o"]="a11_bsag_total")
    declare -A dbTables=(["te_a11"]="te_a11essays")
    declare -A lexTables=(["te_a11"]="tr_a11_xln_")
    declare -A alpha=(["16"]="100" ["32"]="100" ["64"]="100" ["128"]="1000" ["256"]="1000" ["512"]="10000" ["1024"]="10000" ["2048"]="10000")
    dimRedModel=$2
    msgk=$3
    noEval="0"
    if [ "$4" -eq "1" ];
    then
        noEval="1"
    fi
    saveModelCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["te_a11"]}${dimRedModel}${msgk}_w\$tr_a11essays\$clp18_id\$xlne'  --outcome_table tr_variables --outcomes a11_bsag_total --train_regression --model ridge${alpha[${msgk}]} --save_model --picklefile /data/avirinchipur/models/clp${1}_adi/xln_${dimRedModel}_${msgk}.pickle"
else    
    declare -A db=(["d"]="clp19_adi" ["t"]="task_A" ["c"]="user_id" ["o"]="label") 
    declare -A dbTables=(["Ate"]="task_A_test" ["Cte"]="task_Cfil_test" ["Atte"]="task_A_title_test" ["Ctte"]="task_Cfil_title_test")
    declare -A lexTables=(["Ate"]="task_A_xln_" ["Cte"]="task_Cfil_xln_" ["Atte"]="taskAt_xln_" ["Ctte"]="taskCt_xln_")
    dimRedModel=$2
    msgk=$3
    titlek=$4         
    noEval="0"
    if [ "$5" -eq "1" ];
    then
        noEval="1"
    fi    
    saveModelCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_task_A_xln_${dimRedModel}${msgk}_w\$task_A\$user_id\$xlne' 'feat\$cat_taskAt_xln_${dimRedModel}${titlek}_w\$task_A_title\$user_id\$xlne' 'feat\$cat_task_Cfil_xln_${dimRedModel}${msgk}_w\$task_Cfil\$user_id\$xlne' 'feat\$cat_taskCt_xln_${dimRedModel}${titlek}_w\$task_Cfil_title\$user_id\$xlne' --outcome_table task_labels_full --outcomes label --train_classifiers --model lr --save_model --picklefile /data/avirinchipur/models/clp${1}_adi/xln_${dimRedModel}_${msgk}_${titlek}.pickle"         
fi

echo "${saveModelCommand}"

if [ "$noEval" -eq "0" ];
then 
    eval "${saveModelCommand}"
fi


declare -A dimRedModels=(["fa"]="fa" ["pca"]="pca" ["nmf"]="nmf")

resultFile="XLN_${dimRedModel}_${msgk}_${titlek}_test.txt"
echo "$resultFile"


echo "$1_$2_$3"

if [ "${noEval}" -eq "0" ];
then 
    if test -d "~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/"; then 
        echo "Directory Exists"
    else        
        eval "mkdir ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/"
        if test -d "~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/"; then 
            echo "Directory Created"
        fi
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
    
    weightLex="python3  ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${dbTable} -c ${db["c"]} --group_freq_thresh 0 --word_table 'feat\$xlnet_ba_ca_memamiL10co\$${dbTable}\$${db["c"]}\$16to16' --add_lex -l ${lexTable}${dimRedModel}${k} --weighted_lex"
    echo "${weightLex}"

    if [ "${noEval}" -eq "0" ];
    then 
      eval "${weightLex}"
    fi
  
    echo "----------------------------------------------------------------------"
done

if [ "$1" -eq "18" ];
then
    finalCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["te_a11"]}${dimRedModel}${msgk}_w\$te_a11essays\$${db["c"]}\$xlne' --outcome_table te_a11essays_labels --outcomes a11_bsag_total  --predict_regression --model ridge${alpha[${msgk}]}  --load --picklefile /data/avirinchipur/models/clp${1}_adi/xln_${dimRedModel}_${msgk}.pickle > ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/${resultFile}"
else 
    finalCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_task_A_xln_${dimRedModel}${msgk}_w\$task_A_test\$user_id\$xlne' 'feat\$cat_taskAt_xln_${dimRedModel}${titlek}_w\$task_A_title_test\$user_id\$xlne' 'feat\$cat_task_Cfil_xln_${dimRedModel}${msgk}_w\$task_Cfil_test\$user_id\$xlne' 'feat\$cat_taskCt_xln_${dimRedModel}${titlek}_w\$task_Cfil_title_test\$user_id\$xlne' --outcome_table crowd_test_A_label --outcomes label --predict_classifiers  --load --picklefile /data/avirinchipur/models/clp${1}_adi/xln_${dimRedModel}_${msgk}_${titlek}.pickle > ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/XLNet_${dimRedModel}/${resultFile}" 
fi
echo "${finalCommand}" 

if [ "$noEval" -eq "0" ];
then 
    eval "${finalCommand}" 
fi

echo "----------------------------------------------------------------------"

