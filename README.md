[//]: <> (author: @adithya8)

## Readme 


| Number of training samples | Demographic Tasks | Personality Tasks | Mental Health Tasks |
| -------------------------- | :---------------: | :---------------: | :-----------------: |
| 50                         | 16                | 16                | 16                  |
| 100			     | 128		 | 16		     | 22		   |
| 200			     | 512		 | 32		     | 45		   |
| 500			     | 768		 | 64		     | 64		   |
| 1000			     | 768		 | 90 		     | 64		   |

This work is intended to inform researchers in Computational Social Science a simple way to improve the performance of transformer based models. We find that training PCA on transformer representations using the domain data improves the model performance overall, with evidence of handling longer sequences better than other reduction methods.
The table above presents a summary of systematic experiments, recommmending the number of dimensions required for given number of samples in each task domain to achieve the best performance.


The repository contains shell scripts that uses the BERT / XLNet contextual embeddings to produce the results on train (Cross validated) and the test set for three tasks.

- Task 1: CLPsych 2018 task, referred as 18.
- Task 2: CLPsych 2019 task, referred as 19.
- Task 3: FB 20 task, referred as 20.

---

The input arguments for the `BERTDimRedExp.sh` and `XLNetDimRedExp.sh` are given in the table.

|     Task     | Task Identifier |   Dim Red model   | Reduced Dimension 1 | Reduced Dimension 2 |                 Print flag                |
|:------------:|:---------------:|:-----------------:|:-------------------:|:-------------------:|:-------------------------------------------:|
| CLPSych 2018 |        18       | {pca/nmf/nmfr/fa} |         dim         |          -          | 1 - print commands<br>0 - Execute commands |
| CLPSych 2019 |        19       | {pca/nmf/nmfr/fa} |        msg k        |       title k       | 1 - print commands<br>0 - Execute commands |
|     FB 20    |        20       | {pca/nmf/nmfr/fa} |         dim         |          -          | 1 - print commands<br>0 - Execute commands |

Example command:
    
    bash BERTDimRedExp.sh 19 pca 50 20 

The print flag is set to 0 by default so that the execution takes place. 

The input arguments for the `BERTDimRedTest.sh` and `XLNetDimRedTest.sh` are same as the ones 

---

To run the experiments on a particular dataset over multiple k values `Expiter.sh` is executed.

The input arguments for `ExpIter.sh` is given in the table below.

|     Task     | Task Identifier | Contextual Embedding |   Dim Red model   |
|:------------:|:---------------:|----------------------|:-----------------:|
| CLPSych 2018 |        18       | {XLNet/BERTB}        | {pca/nmf/nmfr/fa} |
| CLPSych 2019 |   19 (default)  | {XLNet/BERTB}        | {pca/nmf/nmfr/fa} |
|     FB 20    |        20       | {XLNet/BERTB}        | {pca/nmf/nmfr/fa} |

This runs the dimensionality reduction on the specified contextual embedding of a task for various k values given by the array in the first line of `ExpIter.sh`.

Example command:

    bash ExpIter.sh 19 XLNet pca 

---

The results for cross validation on the train set are written to a text file with the parameters as the file name.

    /results/clp19_adi/BERTb_pca/BERTb_pca_14_6_test.txt
    /results/clp18_adi/XLNet_nmf/XLN_nmf_16_.txt

In the first case, 14 and 6 are the reduced dimensions of message and title respectively. Since there is no title in CLPsych 2018 task, only one numerical param is seen.

*Although the scripts (`BERTDimRedExp.sh` and `XLNetDimRedExp.sh`) takes care of creating the relevant folder for a dimensionality reduction method is not present, the task folder should be created by the user in the current version of script.*

The default folders for storing the result and the model (pickle files) can be modified in the experiment and test scripts. 

---

