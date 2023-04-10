from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from .preprocess import read_csv, clean

if __name__ == '__main__':
	
	# create spark session
	spark = SparkSession\
	.builder\
	.appName("train")\
	.getOrCreate()

	# read file
	df = read_csv(spark,"/home/dea1013/CS643-PA2/TrainingDataset.csv")

	# clean dataframe
	df = clean(df)

	# instantiate model
	rf = RandomForestClassifier(numTrees=10)

	# train model
	model = rf.fit(df)
