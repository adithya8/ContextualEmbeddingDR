#author: @adithya8

if [ "$1" -eq "18" ]; 
then
    declare -A db=(["d"]="clp18_adi" ["t"]="tr_a11essays" ["c"]="clp18_id")
    declare -A dbTables=(["tr_a11"]="tr_a11essays")
    declare -A lexTables=(["tr_a11"]="tr_a11_bertb_")
    folderName="clp18_adi"
    dimRedModel=$2
    msgk=$3
    noEval="0"

    if [ "$4" -eq "1" ];
    then 
      noEval="1"
    fi        
elif [ "$1" -eq "19" ];
then
    declare -A db=(["d"]="clp19_adi" ["t"]="task_A" ["c"]="user_id") 
    declare -A dbTables=(["A"]="task_A" ["C"]="task_Cfil" ["At"]="task_A_title" ["Ct"]="task_Cfil_title")
    declare -A lexTables=(["A"]="task_A_bert_" ["C"]="task_Cfil_bert_" ["At"]="taskAt_bert_" ["Ct"]="taskCt_bert_")
    folderName="clp19_adi"
    dimRedModel=$2
    msgk=$3
    titlek=$4         

    noEval="0"

    if [ "$5" -eq "1" ];
    then 
      noEval="1"
    fi    
else
    declare -A db=(["d"]="fb20_adi" ["t"]="tr_fb" ["c"]="user_id") 
    declare -A dbTables=(["tr_fb"]="tr_fb")
    declare -A lexTables=(["tr_fb"]="tr_fb_bert_")
    folderName="fb20_adi"
    dimRedModel=$2
    msgk=$3    
    noEval="0"

    if [ "$4" -eq "1" ];
    then 
      noEval="1"
    fi        

fi


declare -A dimRedModels=(["fa"]="fa" ["pca"]="pca" ["nmf"]="nmf")

echo "$1_$2_$3"
echo "${noEval}"

resultFile="BERTb_${dimRedModel}_${msgk}_${titlek}.txt"

echo "$resultFile"

if [ "${noEval}" -eq "0" ];
then
    if test -d "~/NLP/ContextualEmbeddingDR/results/${folderName}/BERTb_${dimRedModel}/"; then 
        echo "Directory Exists"
    else        
        eval "mkdir ~/NLP/ContextualEmbeddingDR/results/${folderName}/BERTb_${dimRedModel}/"
        if test -d "~/NLP/ContextualEmbeddingDR/results/${folderName}/BERTb_${dimRedModel}/"; then 
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

    lexTableCr="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${dbTable} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$bert_ba_un_memimaL10co\$${dbTable}\$${db["c"]}\$16to16' --fit_reducer --model ${dimRedModel} --n_components ${k} --reducer_to_lexicon ${lexTable}${dimRedModel}${k}"
    echo "${lexTableCr}"
    
    if [ "${noEval}" -eq "0" ];
    then
        eval "${lexTableCr}"
    fi

    echo "----------------------------------------------------------------------"
    weightLex="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${dbTable} -c ${db["c"]} --group_freq_thresh 0 --word_table 'feat\$bert_ba_un_memimaL10co\$${dbTable}\$${db["c"]}\$16to16' --add_lex -l ${lexTable}${dimRedModel}${k} --weighted_lex"
    echo "${weightLex}"
    
    if [ "${noEval}" -eq "0" ];
    then 
        eval "${weightLex}"
    fi

    echo "----------------------------------------------------------------------"
done
if [ "$1" -eq "18" ];
then
    finalCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["tr_a11"]}${dimRedModel}${msgk}_w\$${db["t"]}\$${db["c"]}\$bert' --outcome_table tr_variables --outcomes a11_bsag_total --combo_test_reg --model ridgehighcv --folds 10 > ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/BERTb_${dimRedModel}/${resultFile}"
elif [ "$1" -eq "19" ];
then 
    finalCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["A"]}${dimRedModel}${msgk}_w\$${db["t"]}\$${db["c"]}\$bert' 'feat\$cat_${lexTables["At"]}${dimRedModel}${titlek}_w\$task_A_title\$user_id\$bert' 'feat\$cat_${lexTables["C"]}${dimRedModel}${msgk}_w\$task_Cfil\$user_id\$bert' 'feat\$cat_${lexTables["Ct"]}${dimRedModel}${titlek}_w\$task_Cfil_title\$user_id\$bert' --outcome_table task_labels_full --outcomes label --nfold_classifiers --model lr  --folds 10 > ~/NLP/ContextualEmbeddingDR/results/clp${1}_adi/BERTb_${dimRedModel}/${resultFile}"
else
    finalCommand="python3 ~/dlatk/dlatk/dlatkInterface.py -d ${db["d"]} -t ${db["t"]} -c ${db["c"]} --group_freq_thresh 0 -f 'feat\$cat_${lexTables["tr_fb"]}${dimRedModel}${msgk}_w\$${db["t"]}\$${db["c"]}\$bert' --outcome_table masterstats_friendratings --outcomes ope con ext agr neu --combo_test_reg --model ridgehighcv --folds 10 > ~/NLP/ContextualEmbeddingDR/results/${folderName}/BERTb_${dimRedModel}/${resultFile}"
fi
echo "${finalCommand}" 
if [ "${noEval}" -eq "0" ];
then
    eval "${finalCommand}" 
fi
echo "----------------------------------------------------------------------"

if [ "${noEval}" -eq "0" ];
then
    if [ "$1" -eq "18" ];
    then
        eval "cat ~/NLP/ContextualEmbeddingDR/results/${folderName}/BERTb_${dimRedModel}/${resultFile} | grep \'r\':"
    elif [ "$1" -eq "19" ];
    then 
        eval "cat ~/NLP/ContextualEmbeddingDR/results/${folderName}/BERTb_${dimRedModel}/${resultFile} | grep \'f1\':"
    else
        eval "cat ~/NLP/ContextualEmbeddingDR/results/${folderName}/BERTb_${dimRedModel}/${resultFile} | grep \'r\':"
    fi
fi