from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def f1(df,model=None):
	"""Evaluate model on a df

	Parameter:
		df (pyspark.sql.DataFrame): Dataframe
		model (pyspark.ml.Model): Model to be evaluated

	Returns:
		pyspark.sql.DataFrame: Read dataframe
	"""
	# transforms input df if model is provided
	if model:
		df = model.transform(df)

	# return f1 score	
	evaluator = MulticlassClassificationEvaluator(metricName='f1Measure')
	return evaluator.evaluate(df)
