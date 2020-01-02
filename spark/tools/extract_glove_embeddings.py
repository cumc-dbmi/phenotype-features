from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql import types as T
from glove import Glove
import argparse

def main(spark, glove_model_path, output_folder):

    glove = Glove.load(glove_model_path)

    dictionary_schema = T.StructType([T.StructField('index', T.IntegerType(), True),
                                      T.StructField('concept_id', T.IntegerType(), True)])

    dictionary_df = spark.createDataFrame([Row(index=k, concept_id=int(v)) for k, v in glove.inverse_dictionary.items()],
                                          dictionary_schema)

    vector_schema = T.StructType([T.StructField('index', T.IntegerType(), True),
                              T.StructField('vector', T.ArrayType(T.DoubleType()), True)])

    vector_df = spark.createDataFrame([Row(index=idx, vector=vector.tolist()) for idx, vector in enumerate(glove.word_vectors)],
                                  vector_schema)

    dictionary_df.join(vector_df, 'index').select('concept_id', 'vector').write.mode('overwrite').parquet(output_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g',
                        '--glove_model_path',
                        dest='glove_model_path',
                        action='store',
                        help='The file path for the glove model',
                        required=True)

    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The output folder that contains the extracted embeddings in the parquet format',
                        required=True)

    ARGS = parser.parse_args()

    spark = SparkSession.builder.appName('Extract glove embeddings').getOrCreate()

    main(spark, ARGS.glove_model_path, ARGS.output_folder)
