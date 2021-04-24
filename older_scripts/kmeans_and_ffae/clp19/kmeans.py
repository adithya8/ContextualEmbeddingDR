import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def run_kmeans(train_data, test_data, k):
    print("################  Clustering  ################")
    km = KMeans(n_clusters=k, random_state=123, max_iter=1000,
                n_init=100)
    print("Running kmeans")
    km.fit(train_data)
    cluster_centers = km.cluster_centers_
    
    print("Train file dimensionality reduction")
    train_distances = euclidean_distances(train_data, cluster_centers)
    print("Test file dimensionality reduction")
    test_distances = euclidean_distances(test_data, cluster_centers)
    
    return train_distances, test_distances



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("train_file_name", help="Train CSV file")
	parser.add_argument("test_file_name", help="Test CSV file")
	parser.add_argument("k", help="number of clusters")
	
	args = parser.parse_args()

	k = int(args.k)
	train_file = args.train_file_name
	test_file = args.test_file_name
	train_data = pd.read_csv(train_file)
	test_data = pd.read_csv(test_file)
	train_data_wo_group_ids = train_data.iloc[:, 1:]
	train_group_ids = train_data.iloc[:, 0]
	test_data_wo_group_ids = test_data.iloc[:,1:]
	test_group_ids = test_data.iloc[:, 0]

	train_distances, test_distances = run_kmeans(train_data_wo_group_ids, test_data_wo_group_ids, k)


	train_distances = pd.DataFrame(train_distances)
	train_distances.insert(0, 'group_id', train_group_ids)
	train_distances_aslist = []
	idx = 0
	for i in range(train_distances.shape[0]):
		for j in range(1, train_distances.shape[1]):
			idx += 1
			train_distances_aslist.append([idx, train_distances.iloc[i,0], "COMPONENT_" + str(train_distances.columns[j]),train_distances.iloc[i,j], train_distances.iloc[i,j]])      

	train_reduced = pd.DataFrame.from_records(train_distances_aslist, index = None)

	print("Creating reduced train csv file...")
	print("Converting","data from",train_data.shape,"to","("+str(train_data.shape[0])+","+str(k)+")")
 
	train_reduced.columns=["id", "group_id", "feat", "value", "group_norm"]
	train_reduced.to_csv(train_file.split(".")[0] + "_kmeans.csv", index = False)
 
	test_distances = pd.DataFrame(test_distances)
	test_distances.insert(0, 'group_id', test_group_ids)
	test_distances_aslist = []
	idx = 0
	for i in range(test_distances.shape[0]):
		for j in range(1, test_distances.shape[1]):
			idx += 1
			test_distances_aslist.append([idx, test_distances.iloc[i,0], "COMPONENT_" + str(test_distances.columns[j]), test_distances.iloc[i,j], test_distances.iloc[i,j]])      

	test_reduced = pd.DataFrame.from_records(test_distances_aslist, index = None)

	print("Creating reduced test csv file...")
	print("Converting","data from",test_data.shape,"to","("+str(test_data.shape[0])+","+str(k)+")")
 
	test_reduced.columns=["id", "group_id", "feat", "value", "group_norm"]
	test_reduced.to_csv(test_file.split(".")[0] + "_kmeans.csv", index = False)
