This directory contains models for reducing tranformer-based embeddings to smaller numbers of dimensions as described in [Ganesan et al. (NAACL 2021)](aclanthology.org/2021.naacl-main.357/). 

## What dimension reduction model should I use for my task? 
**We recommend the tool at the top of our [blog post](adithya8.github.io/blog/paper/2021/04/15/Empirical-Evaluation.html)** which will provide a direct link based on the characteristics of your training task. The information below is intended for the more dedicated users that want access to mamy models. 

## Available Models

Although we trained models for a few domains of corpora, we recommend those using RoBERTA-base layer 11 and trained over the Facebook corpora as they were the largest and seemed to generalize best to new situations. These include:
 * `rpca_roberta_16_D_20` - Transformation from 768 dimensions of Layer 11 (second-to-last) from roberta-base to **16** dimensions. <br /> [csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_16_D_20.csv), [pickle](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_16_D_20.pickle)
 * `rpca_roberta_32_D_20` - Transformation from 768 dimensions of Layer 11 (second-to-last) from roberta-base to **32** dimensions. <br /> [csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_32_D_20.csv), [pickle](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_32_D_20.pickle)
 * `rpca_roberta_64_D_20` - Transformation from 768 dimensions of Layer 11 (second-to-last) from roberta-base to **64** dimensions. <br /> [csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_64_D_20.csv), [pickle](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_64_D_20.pickle)
 * `rpca_roberta_128_D_20` - Transformation from 768 dimensions of Layer 11 (second-to-last) from roberta-base to **128** dimensions. <br /> [csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_128_D_20.csv), [pickle](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_128_D_20.pickle)
 * `rpca_roberta_256_D_20` - Transformation from 768 dimensions of Layer 11 (second-to-last) from roberta-base to **256** dimensions. <br /> [csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_256_D_20.csv), [pickle](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_256_D_20.pickle)
 * `rpca_roberta_512_D_20` - Transformation from 768 dimensions of Layer 11 (second-to-last) from roberta-base to **512** dimensions. <br /> [csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_512_D_20.csv), [pickle](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_512_D_20.pickle)
 * `rpca_roberta_768_D_20` - Transformation from 768 dimensions of Layer 11 (second-to-last) from roberta-base to **768** dimensions. This 768 version does not provide any reduction from layer 11's hidden state size, but we still find PCA sometimes provides benefits in terms of insuring orthogonal dimension input to a model. <br /> [csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_768_D_20.csv), [pickle](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/rpca_roberta_768_D_20.pickle)

**Note**: If you download the csvs, please be sure to get the [scaler.csv](https://github.com/adithya8/ContextualEmbeddingDR/blob/master/models/fb20/scalar.csv) which is needed to transform your dimensions before reducing (see Model Usage below). 

## Model formats
The models are available in two formats:

 * `.csv` - A comma-seprated value format storing the linear transformation matrix of size `K X 768` where `K` is the low dimension.
 * `.pickle` - A pickle of the [Scikit-learn Decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) modules. 

## Model Usage

The model files contain the matrix to reduce the dimensions of these user representations and this README explains how to use these models.

There are two steps to applying the models:

 1. **Normalize**: Standardize the dimensions before reducing based on the mean and standard deviation provided with the model. This is the same for all reduction sizes. 
 2. **Reduce Transform**: Apply the linear transform from 768 to the low dimensions. This is different for each reduction size. 

### Input format

All models assume you have an input matrix of `N_observations X 768`, where `N_observations` is the training set size. The goal is produce `N_obsevations X K` output where `K` is the lower dimensional represetnation from the model. 

*Aggregating to user-level.* In many situations, one has multiple documents/posts/messages per individual. To form user representation, the message representations of each user is averaged. 

### Using CSVs through python

Here is an example for how to use the models in CSV format:

```py
def transform(user_emb):
	#input:
	#   user_emb: numpy matrix of N_observations X 768  -- matrix of average RoBERTA layer 11 per user. 
	#output:
	#   transformed_user_emb: numpy matrix of N_observations X K -- low dimensional user representation. 
	import numpy as np
	
	#Step 1: Normalize: 
	scalar = np.loadtxt("scalar.csv", delimiter=",")
	#shape: (2, 768); 1st row -> mean; 2nd row -> std
	user_emb = (user_emb - scalar[0]) / scalar[1]
	
	#Step 2: Reduce Transform
	model = np.loadtxt("rpca_roberta_K_D_20.csv", delimiter=",") #replace K
	#shape of model: (K, 768)
	transformed_user_emb = np.dot(user_emb, model.T)
	return transformed_user_emb
```
### Using pickle files through python

These pickle files are composed of a [Scikit-learn Decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition). To apply the learnt reduction, you can unpickle the model and run transform() method on the user embeddings.
Here is an example showing how:

```py
def transform(user_emb):
	#input:
	#   user_emb: numpy matrix of N_observations X 768  -- matrix of average RoBERTA layer 11 per user. 
	#output:
	#   transformed_user_emb: numpy matrix of N_observations X K -- low dimensional user representation. 
	import pickle 
	with open("model.pickle", "rb") as f:
		scalar = pickle.load(f)['scalers']['noOutcome']
		model = pickle.load(f)['clusterModels']['noOutcome']
	user_emb = scalar.transform(user_emb)
	transformed_user_emb = model.transform(user_emb)
	return transformed_user_emb
```
### Using pickle files through DLATK

The message data is composed of the user id (user_id), message id (message_id), the message field and the outcome field(s). The user embeddings are generated by averaging the transformer representation of all the messages from a user. 

If the user embeddings have been generated using [DLATK](https://github.com/DLATK/DLATK/) by following the commands [here](https://github.com/adithya8/ContextualEmbeddingDR#commands-to-extract-dimension-reduced-tables-using-a-specific-method), then you can use these pickle files directly by using the following command:

	python dlatkInterface.py -d {database-name} -t {table-name} -g {group-name} -f {user-embeddings-table-name} \
	--transform_to_feats {dimred-table-name} --load --pickle {path-to-pickle-file}

## Model Description

The models made available are pre-trained to reduce 768 dimensions of roberta-base using 3 datasets from different domains: Facebook (D_20), CLPsych 2019 (D_19), and CLPsych 2018 (D_18).
D_20 dataset contains facebook posts of 55k users, while the D_19 has reddit posts from 496 users on r/SuicideWatch and D_18 contains essays written by approx 10k children. To know more about these datasets, refer to Section 3 in our paper.

Model files have been named by following this nomenclature: {method}\_{embedding}\_{dimensions}\_{dataset}.{extension}

	model: The reduction method
	embedding: The transformer model used
	dimensions: Number of reduced dimensions
	dataset: One of the three identifiers - D_20, D_19 or D_18.
	extension: csv or pickle

Email us in case you would like to use a model from our paper that is not made available here.  

---

You can cite our work with:
	
        @inproceedings{v-ganesan-etal-2021-empirical,
        title = "Empirical Evaluation of Pre-trained Transformers for Human-Level {NLP}: The Role of Sample Size and Dimensionality",
        author = "V Ganesan, Adithya  and Matero, Matthew  and Ravula, Aravind Reddy  andVu, Huy  and Schwartz, H. Andrew",
        booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
        month = jun,
        year = "2021",
        address = "Bangkok, Thailand",
        publisher = "Association for Computational Linguistics",
        url = "aclanthology.org/2021.naacl-main.357/",
        pages = "4515--4532"}
