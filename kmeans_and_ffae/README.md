### Folder structure

    .
    ├── clp18                    
    │   ├── autoencoder.py                       # Runs autoencoder
    │   ├── clp18_bert_ffae_results.sh           # Generates all the results on bert features using ffae
    │   └── clp18_xlnet_ffae_results.sh          # Generates all the results on xlnet features using ffae
    ├── clp19                    # Test files (alternatively `spec` or `tests`)
    │   ├── autoencoder.py                       # Runs autoencoder
    │   ├── clp18_bert_ffae_results.sh           # Generates all the results on bert features using ffae
    │   ├── clp18_xlnet_ffae_results.sh          # Generates all the results on xlnet features using ffae
    │   ├── autoencoder.py                       # Runs autoencoder
    │   ├── kmeans.py                            # Kmeans algorithm implemented here
    │   ├── xlnet_kmeans.sh                      # Runs kmeans on the four tables(task_A, task_A_title, task_Cfil, task_Cfil_title)
    │   ├── clp19_bert_kmeans_results.sh         # Generates all the results on bert features using kmeans
    │   └── clp19_xlnet_kmeans_results.sh        # Generates all the results on xlnet features using kmeans
    ├── fb20                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit test
