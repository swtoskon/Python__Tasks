#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:14:05 2020

@author: swtoskon
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StructType,StructField, StringType, DecimalType, IntegerType , DecimalType , TimestampType 

spark = SparkSession.builder\
                .master("yarn-client")\
                .appName("csv_to_parquet")\
                .getOrCreate()

schema = StructType() \
      .add("timestamp",TimestampType(),True) \
      .add("amount",DecimalType(6,4),True) \
      .add("channel",StringType(),True) 
      
df=spark.read.csv("hdfs://localhost:9000/Hadoop_File/data.csv",schema=schema,sep=',')
print(df.count())
#print(df.show())
filtered_data = df.filter(~(f.col('timestamp').isNull()))
filtered_data = filtered_data.filter(~(f.col('amount').isNull()))
filtered_data = filtered_data.filter(~(f.col('channel').isNull())  )
filtered_data = filtered_data.filter(~(f.col('channel').startswith('*'))) 
#filtered_data = filtered_data.filter(~(f.col('channel').startswith('#'))) 
print(filtered_data.count())                       
#filtered_data.show()
filtered_data.write.mode('overwrite').parquet("hdfs://localhost:9000/Hadoop_File/output.parquet")
parquetFile = spark.read.parquet("hdfs://localhost:9000/Hadoop_File/output.parquet")
#parquetFile.printSchema()
parquetFile.createOrReplaceTempView("parquetFile")
#example query
query = spark.sql("SELECT * FROM parquetFile WHERE amount > 1")
query.show()
print(query.count())