from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from preprocess import read_csv, clean

if __name__ == '__main__':
	
	# create spark session
	spark = SparkSession\
	.builder\
	.appName("train")\
	.getOrCreate()

	# read file
	df = read_csv(spark,"s3://dilip-anand-cs643-pa2/TrainingDataset.csv")

	# clean dataframe
	df = clean(df)

	# instantiate model
	rf = RandomForestClassifier(numTrees=40, maxDepth=5, impurity='entropy')

	# train model
	model = rf.fit(df)

	# export
	model.write().overwrite().save("s3://dilip-anand-cs643-pa2/model")
