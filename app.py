import argparse

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from preprocess import read_csv, clean
from evaluate import f1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path')
	args = parser.parse_args()

	# create spark session
	spark = SparkSession\
	.builder\
	.appName("train")\
	.getOrCreate()

	# read file
	df = read_csv(spark,args.path)

	# clean dataframe
	df = clean(df)

	# model
	model = RandomForestClassifier.load("model")

	# get f1 score
	score = f1(df,model)

	print("F1 Score: {}".format(score))
