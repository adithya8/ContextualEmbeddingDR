#! /bin/bash
msg=(14 36 71 143 286 493)
title=(6 14 29 57 114 198)
arraylength=${#msg[@]}
for (( i=1; i<${arraylength}+1; i++ ));
do
    echo "----------------------------------------------------------------------"
    echo "Running BERT for dim red ${msg[$i-1]} ${title[$i-1]}"
    echo "----------------------------------------------------------------------"
    sh /users2/aravula/clp19_results/bert_kmeans.sh ${msg[$i-1]} ${title[$i-1]}
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_A\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_A\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_Cfil\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_Cfil\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_A_title\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_A_title\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_Cfil_title\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_Cfil_title\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_A_test\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_A_test\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_Cfil_test\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_Cfil_test\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_A_title_test\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_A_title_test\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/clp19_results/feat\$bert_ba_un_memimaL10co\$task_Cfil_title_test\$user_id\$16to16_kmeans.csv clp19_adi feat\$cat_clp19_task_At_bert_km\$task_Cfil_title_test\$user_id '(id bigint(16) unsigned, group_id bigint(20), feat varchar(12), value int(11), group_norm double)' 1

    /users2/aravula/dlatk/dlatkInterface.py -d clp19_adi -t task_A -c user_id --group_freq_thresh 0 -f 'feat$cat_clp19_task_At_bert_km$task_A$user_id' 'feat$cat_clp19_task_At_bert_km$task_A_title$user_id' 'feat$cat_clp19_task_At_bert_km$task_Cfil$user_id' 'feat$cat_clp19_task_At_bert_km$task_Cfil_title$user_id'   --outcome_table task_labels_full --outcomes label --train_classifiers --model lr --save_model --picklefile /data/aravula/bert_pickles/bert_kmeans_${msg[$i-1]}_${title[$i-1]}_clp19.pickle
    /users2/aravula/dlatk/dlatkInterface.py -d clp19_adi -t task_A -c user_id --group_freq_thresh 0 -f 'feat$cat_clp19_task_At_bert_km$task_A_test$user_id' 'feat$cat_clp19_task_At_bert_km$task_A_title_test$user_id' 'feat$cat_clp19_task_At_bert_km$task_Cfil_test$user_id' 'feat$cat_clp19_task_At_bert_km$task_Cfil_title_test$user_id' --outcome_table crowd_test_A_label --outcomes label --predict_classifiers --load --picklefile /data/aravula/bert_pickles/bert_kmeans_${msg[$i-1]}_${title[$i-1]}_clp19.pickle > /data/aravula/bert_pickles/bert_kmeans_${msg[$i-1]}_${title[$i-1]}_clp19.txt

done