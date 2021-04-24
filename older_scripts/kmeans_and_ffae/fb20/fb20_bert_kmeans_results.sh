#! /bin/bash
msg=(16 32 64 128 256)
arraylength=${#msg[@]}
for (( i=1; i<${arraylength}+1; i++ ));
do
    echo "----------------------------------------------------------------------"
    echo "Running msg ${msg[$i-1]}"
    echo "----------------------------------------------------------------------"
    sh /users2/aravula/fb20_results/bert_kmeans.sh ${msg[$i-1]}
    
    
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/fb20_results/feat\$bert_ba_un_memimaL10co\$tr_fb\$user_id\$16to16_kmeans.csv fb20_adi feat\$bert_ba_un_memimaL10co_km\$tr_fb\$user_id '(id bigint(16) unsigned, group_id bigint(20) unsigned, feat varchar(12), value double, group_norm double)' 1
    
    
    python /users2/aravula/tools/usefulScripts/csv2mySQL.py /users2/aravula/fb20_results/feat\$bert_ba_un_memimaL10co\$te_fb\$user_id\$16to16_kmeans.csv fb20_adi feat\$bert_ba_un_memimaL10co_km\$te_fb\$user_id '(id bigint(16) unsigned, group_id bigint(20) unsigned, feat varchar(12), value double, group_norm double)' 1
    

    /users2/aravula/dlatk/dlatkInterface.py -d fb20_adi -t tr_fb -c user_id --group_freq_thresh 0 -f 'feat$bert_ba_un_memimaL10co_km$tr_fb$user_id' --outcome_table masterstats_friendratings --outcomes ope con ext agr neu --combo_test_reg --model ridgehighcv --folds 10 > /data/aravula/fb20/bert_kmeans_${msg[$i-1]}_fb20

done
