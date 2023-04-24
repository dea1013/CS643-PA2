from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType


def read_csv(spark,path):
	"""Read dataframe from CSV

	Parameter:
		spark (pyspark.sql.SparkSession): Spark session
		path (String): Path to CSV file 

	Returns:
		pyspark.sql.DataFrame: Read dataframe
	"""
    
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
	.load(path)
        
	return df



def clean(df):
    """Clean dataframe

	Parameter:
		df (pyspark.sql.DataFrame): Dataframe

	Returns:
		pyspark.sql.DataFrame: Clean dataframe
	"""
    
    # rename columns and remove quotes
    for col in df.columns:
        df = df.withColumnRenamed(col, col.replace('"',''))
    df = df.withColumnRenamed('quality', 'label')

    # format data in form of label and features
    feature_cols = df.columns[:-1]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features', handleInvalid = "skip")
    df = assembler.transform(df).select('label','features')

    return df


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
