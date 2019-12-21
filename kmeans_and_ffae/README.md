### Folder structure

    .
    ├── clp18                    
    │   ├── autoencoder.py                       # Runs autoencoder
    │   ├── clp18_bert_ffae_results.sh           # Generates all the results on bert features using ffae
    │   └── clp18_xlnet_ffae_results.sh          # Generates all the results on xlnet features using ffae
    ├── clp19                    
    │   ├── autoencoder.py                       # Runs autoencoder
    │   ├── clp18_bert_ffae_results.sh           # Generates all the results on bert features using ffae
    │   ├── clp18_xlnet_ffae_results.sh          # Generates all the results on xlnet features using ffae
    │   ├── autoencoder.py                       # Runs autoencoder
    │   ├── kmeans.py                            # Kmeans algorithm implemented here
    │   ├── xlnet_kmeans.sh                      # Runs kmeans on the four tables(task_A, task_A_title, task_Cfil, task_Cfil_title)
    │   ├── clp19_bert_kmeans_results.sh         # Generates all the results on bert features using kmeans
    │   └── clp19_xlnet_kmeans_results.sh        # Generates all the results on xlnet features using kmeans
    ├── fb20                    
    │   ├── kmeans.py                            # Kmeans algorithm implemented here
    │   ├── xlnet_kmeans.sh                      # Runs kmeans on the tables
    │   ├── fb20_bert_kmeans_results.sh          # Generates all the results on bert features using kmeans
    │   └── fb20_xlnet_kmeans_results.sh         # Generates all the results on xlnet features using kmeans


Tasks(task)<br/>
    1. clp18 <br/>
    2. clp19<br/>
    3. fb20<br/>

Contextual Embeddings(ce)<br/>
    1. xlnet<br/>
    2. bert<br/>
    <br/>
Dimensionality Reduction Techniques(dr)<br/>
    1. kmeans<br/>
    2. ffae<br/>
    
In each of the directory there is a file named(in the format) task_ce_dr_results.sh
Example command:

    bash task_ce_dr_results.sh
    
Upon running the above command it will generate .pickle and .txt (containing results of testing) in the /data/aravula/dr_pickles/ folder with names ce_dr_messagereducedsize_titlereducedsize_task.pickle and ce_dr_messagereducedsize_titlereducedsize_task.txt respectively.
