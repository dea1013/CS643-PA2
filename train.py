from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

if __name__ == '__main__':

        # create spark session
        spark = SparkSession\
                .builder\
                .appName("train")\
                .getOrCreate()
        
        # schema
        schema = StructType([
                StructField('""""fixed acidity""""', DoubleType()),
                StructField('""""volatile acidity""""', DoubleType()),
                StructField('""""citric acid""""', DoubleType()),
                StructField('""""residual sugar""""', DoubleType()),
                StructField('""""chlorides""""', DoubleType()),
                StructField('""""free sulfur dioxide""""', IntegerType()),
                StructField('""""total sulfur dioxide""""', IntegerType()),
                StructField('""""density""""', DoubleType()),
                StructField('""""pH""""', DoubleType()),
                StructField('""""sulphates""""', DoubleType()),
                StructField('""""alcohol""""', DoubleType()),
                StructField('""""quality""""', IntegerType()),
        ])

        # read file
        df = spark.read.format("csv")\
        .option('delimiter', ';')\
        .option('header', True)\
        .schema(schema)\
        .load("/home/dea1013/CS643-PA2/TrainingDataset.csv")

        # create features and label column
        feature_cols = df.columns[:-1]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        df = assembler.transform(df).select('""""quality""""','features')
        df = df.withColumnRenamed('""""quality""""', 'label')

        # instantiate model
        lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

        # train model
        lrModel = lr.fit(df)

        print(lrModel.summary)
