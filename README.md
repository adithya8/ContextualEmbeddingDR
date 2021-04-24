# **Empirical Evaluation of Pre-trained Transformers for Human-Level NLP: The Role of Sample Size and Dimensionality**

## *Adithya V Ganesan, 112683104*

----

The FB dataset detailed in the report was used for this project. The data was dumped in MySQL in order to conduct all the experiments using [DLATK](https://github.com/dlatk/dlatk), a language toolkit. This README file contains the command logs to collect the results for the experiments. All the commands except for the transformer embedding generation was executed by switching to the "dev" branch. The said embedding generation commands were executed checkingout to "dev-transformers" branch.

The messages for domain data was stored in table named D_20 and task data in a table named T_20. The outcomes (age, gen, ext, ope) were stored in a table named 20_outcomes. The database will be referred as "db".

---- 

### **Commands to extract RoBERTa embeddings:**

The DLATK first requires to tokenize the messages before generating the embeddings: 

    python3 dlatkInterface.py -d db -t table_name -c user_id --add_sent_tokenized

The embedding generation command:

    CUDA_VISIBLE_DEVICES=0 python3 dlatkInterface.py -d db -t table_name -c user_id --add_emb --emb_model roberta-base --emb_layers 11 --emb_msg_aggregation mean --batch_size 30

table_name = {D_20, T_20}

----

### **Commands to extract dimension reduced tables using a specific method:**

The DLATK command to extract the dimension reduction is done in two steps as explained in the report. The first step involved learning the reduction on the domain data and storing the learnt model in a pickle file:

    python3 dlatkInterface.py -d db -t table_name -c user_id --group_freq_thresh 1000 -f 'feat$roberta_ba_meL11con$table_name$user_id$16to16' --model {dimred_model} --fit_reducer --k 128 --save_model --picklefile dimred_model_128.pickle

The number of dimensions to reduce to (components) can be changed by altering the argument of `--k`

The dimred_model here could be `pca, nmf, fa, ae` (for non linear auto-encoders). For ae, it is expected to prefix the command with CUDA_VISIBLE_DEVICES env variable to specify GPU.  

The second step would be applying this learnt reduction model on the task data to generate the reduced representations.

    python3 dlatkInterface.py -d db -t table_name -c user_id --group_freq_thresh 1000 -f 'feat$roberta_ba_meL11con$table_name$user_id$16to16' --transform_to_feats {dimred_table_name} --load --picklefile dimred_model_128.pickle

The name of the table to stored the dimension reduced representations is given in pace of dimred_table_name. 

-----

### **Commands to perform bootstrapped training and evaluation:**

The commands to perform bootstrapped training followed by evaluation for regression task is given by:

    python3 dlatkInterface.py -d db -t task_table_name -c user_id --group_freq_thresh 1000 -f '{feat_table_name}' --outcome_table 20_outcomes --outcomes age ext ope --train_reg --model ridgehighcv --train_bootstraps 10 --where 'r10pct_test_fold is NOT NULL' --train_bootstraps_ns 50 100 200 500 1000 --no_standardize --save_models --picklefile reg_model_{feat_table_name}.pickle

The feat table name is either the raw embeddings table name or the dimension reduced feature table name. The regression outcomes are listed in the `--outcomes` flag. The number of times to perform the bootstrapping is specified in `--train_bootstraps` flag and the sample sizes for bootstrapping is specifed in `--train_bootstrap_ns` flag. Finally the ridge model is saved as a pickle file. 

    python3 ~/NLP/dlatk/dlatkInterface.py -d db-t table_name -c user_id --group_freq_thresh 1000 -f '{feat_table_name}' --outcome_table 20_outcomes --outcomes age ext ope --predict_reg --where 'facet_fold = 1' --load --picklefile reg_model_{feat_table_name}.pickle > output.txt

This command would store the evaluation result for the ten runs in output.txt. 

For classification task the commands have a slight variation. The outcomes fag is changed to appropriate categorical column name. The `--train_reg` and `--predict_reg` are changed to `--train_classifiers` and `--predict_classifiers` respectively. 

----
