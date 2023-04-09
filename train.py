from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == '__main__':

        # create spark session
        spark = SparkSession\
                .builder\
                .appName("train")\
                .getOrCreate()

        # schema
        schema = StructType([
                StructField('"""""fixed acidity""""', DoubleType(), True),
                StructField('""""volatile acidity""""', DoubleType(), True),
                StructField('""""citric acid""""', DoubleType(), True),
                StructField('""""residual sugar""""', DoubleType(), True),
                StructField('""""chlorides""""', DoubleType(), True),
                StructField('""""free sulfur dioxide""""', IntegerType(), True),
                StructField('""""total sulfur dioxide""""', IntegerType(), True),
                StructField('""""density""""', DoubleType(), True),
                StructField('""""pH""""', DoubleType(), True),
                StructField('""""sulphates""""', DoubleType(), True),
                StructField('""""alcohol""""', DoubleType(), True),
                StructField('""""quality"""""', IntegerType(), True),
        ])

        # read file
        df = spark.read.format("csv")\
        .option('delimiter', ';')\
        .option('header', True)\
        .schema(schema)\
        .load("/home/dea1013/CS643-PA2/TrainingDataset.csv")

        # create features and label column
        for col in df.columns:
                df = df.withColumnRenamed(col, col.replace('"',''))
        df = df.withColumnRenamed('quality', 'label')
        feature_cols = df.columns[:-1]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features', handleInvalid = "skip")
        df = assembler.transform(df).select('label','features')

        # instantiate model
        rf = RandomForestClassifier(numTrees=10)

        # train model
        model = rf.fit(df)

        # predictions
        predictions = model.transform(df)

        # evaluation
        evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
        accuracy = evaluator.evaluate(predictions)
        print('Accuracy:', accuracy)
