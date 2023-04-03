from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

if __name__ == '__main__':

        spark = SparkSession\
                .builder\
                .appName("train")\
                .getOrCreate()

        df = spark.read.format("csv")\
        .option('header', True)\
        .option('inferSchema', True)\
        .load("/home/dea1013/CS643-PA2/TrainingDataset.csv")

        lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

        lrModel = lr.fit(df)

        print(lrModel.summary)
