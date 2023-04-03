from pyspark.ml.classification import LogisticRegression

dataset = spark.read.format("csv").\
option('header', True).\
option('inferSchema', True).\
load("/home/dea1013/TrainingDataset.csv")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

lrModel = lr.fit(dataset)

print(lrModel.summary)