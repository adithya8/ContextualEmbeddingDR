## **Model Usage**

### **Using pickle files through python**

These pickle files are composed of a sckit learn decomposition class. Hence, to apply the learnt reduction, you can unpickle the model and run transform() method on the user embeddings.
Here is an example showing how:

	import pickle 
	with open("model.pickle", "rb") as f:
		model = pickle.load(f)["clusterModels"]["noOutcome"]
	#user embeddings are stored in a variable calles user_emb
	transformed_user_emb = model.transform(user_emb)

### **Using pickle files through DLATK**

If the user embeddings have been generated using [DLATK](https://github.com/DLATK/DLATK/) by following the commands [here](https://github.com/adithya8/ContextualEmbeddingDR#commands-to-extract-dimension-reduced-tables-using-a-specific-method), then you can use these pickle files directly by using the following command:

	python dlatkInterface.py -d {database-name} -t {table-name} -g {group-name} -f {user-embeddings-table-name} \
	--transform_to_feats {dimred-table-name} --load --pickle {path-to-pickle-file}


### **Using CSVs through python**

If you are using the CSVs, here is an example for how to use it:

	import numpy as np
	model = np.loadtxt("model.csv", delimiter=",")
	#shape of model: (x, 768)
	transformed_user_emb = np.dot(user_emb, model.T)


## **Model Description**

The models made available are pre-trained to reduce 768 dimensions of roberta-base using 3 datasets from different domains: Facebook (D_20), CLPsych 2019 (D_19), and CLPsych 2018 (D_18).
D_20 dataset contains facebook posts of 55k users, while the D_19 has reddit posts from 496 users on r/SuicideWatch and D_18 contains essays written by approx 10k children. To know more about these datasets, refer to Section 3 in our paper.

Model files have been named by following this nomenclature: {method}\_{embedding}\_{dimensions}\_{dataset}.{extension}

	model: The reduction method
	embedding: The transformer model used
	dimensions: Number of reduced dimensions
	dataset: One of the three identifiers - D_20, D_19 or D_18.
	extension: csv or pickle

Email us in case you would like to use a model from our paper that is not made available here.  
