## **Model usage**

---

### **Using pickle files through python**

These pickle files are composed of a sckit learn decomposition class. Hence, to apply the learnt reduction, you can unpickle the model and run transform() method on the user embeddings.
Here is an example showing how:

	import pickle 
	with open("model.pickle", "rb") as f:
		model = pickle.load(f)["clusterModels"]["noOutcome"]
	#user embeddings are stored in a variable calles user_emb
	user_emb = model.transform(user_emb)

### **Using pickle files through DLATK**

If the user embeddings have been generated using [DLATK](github.com/DLATK/DLATK/) by following the commands [here](https://github.com/adithya8/ContextualEmbeddingDR#commands-to-extract-dimension-reduced-tables-using-a-specific-method), then you can use these pickle files directly by using the following command:

	python dlatkInterface.py -d {database-name} -t {table-name} -g {group-name} -f {user-embeddings-table-name} --transform_to_feats {dimred-table-name} --load --pickle {path-to-pickle-file}


