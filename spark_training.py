from pyspark.sql import SparkSession
import pandas as pd
import os
from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.ml as ml
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from IPython.display import display
import glob
from pyspark import storagelevel
base_dir="/data1/r08922010/recsys2021"

files_nums=50



from pyspark.sql import SparkSession
import pandas as pd
import os
from pyspark.sql.types import *
from pyspark.ml.feature import *
import pyspark.ml as ml
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from IPython.display import display
import glob
from pyspark import storagelevel
base_dir="/data1/r08922010/recsys2021"
​
files_nums=50
​
​
​
gpu_amount=3
concurrentGpuTasks=10
spark = SparkSession.builder\
.master("spark://hpcuda2:7077")\
.config("spark.kryoserializer.buffer.max", "2047m")\
.config("spark.sql.files.maxPartitionBytes", "2047m")\
.config("spark.local.dir", "/data1/tmp_spark")\
.config("dfs.blocksize", "2047m")\
.config("spark.executor.cores", 25)\
.config("spark.executor.memory","100g")\
.config("spark.driver.extraClassPath","/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar:/opt/sparkRapidsPlugin/xgboost4j_3.0-1.3.0-0.1.0.jar:/opt/sparkRapidsPlugin/xgboost4j-spark_3.0-1.3.0-0.1.0.jar")\
.config("spark.executor.extraClassPath","/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar:/opt/sparkRapidsPlugin/xgboost4j_3.0-1.3.0-0.1.0.jar:/opt/sparkRapidsPlugin/xgboost4j-spark_3.0-1.3.0-0.1.0.jar")\
.config("spark.plugins", "com.nvidia.spark.SQLPlugin")\
.config("spark.worker.resource.gpu.amount", gpu_amount)\
.config("spark.executor.resource.gpu.amount", 1)\
.config("spark.rapids.sql.concurrentGpuTasks",concurrentGpuTasks)\
.config("spark.task.resource.gpu.amount", 1/concurrentGpuTasks)\
.config("spark.rapids.memory.pinnedPool.size", "100G")\
.config("spark.rapids.memory.gpu.maxAllocFraction", 0.7)\
.config("spark.rapids.memory.gpu.allocFraction", 0.7)\
.config("spark.locality.wait", "0s")\
.config("spark.worker.resource.gpu.discoveryScript","/opt/sparkRapidsPlugin/getGpusResources.sh" )\
.config("spark.sql.shuffle.partitions",files_nums)\
.config("spark.sql.files.maxPartitionBytes",'512m')\
.config("spark.rapids.sql.incompatibleOps.enabled",True)\
.getOrCreate()
spark.sparkContext.addPyFile("/opt/sparkRapidsPlugin/xgboost4j-spark_3.0-1.3.0-0.1.0.jar")
from sparkxgb import XGBoostClassifier, XGBoostClassificationModel

print(spark.sparkContext.uiWebUrl)

gpu_amount=3
concurrentGpuTasks=10
spark = SparkSession.builder\
.master("spark://hpcuda2:7077")\
.config("spark.kryoserializer.buffer.max", "2047m")\
.config("spark.sql.files.maxPartitionBytes", "2047m")\
.config("spark.local.dir", "/data1/tmp_spark")\
.config("dfs.blocksize", "2047m")\
.config("spark.executor.cores", 25)\
.config("spark.executor.memory","100g")\
.config("spark.driver.extraClassPath","/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar:/opt/sparkRapidsPlugin/xgboost4j_3.0-1.3.0-0.1.0.jar:/opt/sparkRapidsPlugin/xgboost4j-spark_3.0-1.3.0-0.1.0.jar")\
.config("spark.executor.extraClassPath","/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar:/opt/sparkRapidsPlugin/xgboost4j_3.0-1.3.0-0.1.0.jar:/opt/sparkRapidsPlugin/xgboost4j-spark_3.0-1.3.0-0.1.0.jar")\
.config("spark.plugins", "com.nvidia.spark.SQLPlugin")\
.config("spark.worker.resource.gpu.amount", gpu_amount)\
.config("spark.executor.resource.gpu.amount", 1)\
.config("spark.rapids.sql.concurrentGpuTasks",concurrentGpuTasks)\
.config("spark.task.resource.gpu.amount", 1/concurrentGpuTasks)\
.config("spark.rapids.memory.pinnedPool.size", "100G")\
.config("spark.rapids.memory.gpu.maxAllocFraction", 0.7)\
.config("spark.rapids.memory.gpu.allocFraction", 0.7)\
.config("spark.locality.wait", "0s")\
.config("spark.worker.resource.gpu.discoveryScript","/opt/sparkRapidsPlugin/getGpusResources.sh" )\
.config("spark.sql.shuffle.partitions",files_nums)\
.config("spark.sql.files.maxPartitionBytes",'512m')\
.config("spark.rapids.sql.incompatibleOps.enabled",True)\
.getOrCreate()
spark.sparkContext.addPyFile("/opt/sparkRapidsPlugin/xgboost4j-spark_3.0-1.3.0-0.1.0.jar")
from sparkxgb import XGBoostClassifier, XGBoostClassificationModel
​
print(spark.sparkContext.uiWebUrl)
​
http://Milgram.csie.ntu.edu.tw:4040
# /opt/spark/bin/pyspark \
#        --master spark://hpcuda2:7077 \
#        --conf spark.executor.extraClassPath=/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar\
#        --conf spark.driver.extraClassPath=/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar \
#        --conf spark.rapids.memory.gpu.allocFraction=0.01 \
#        --conf spark.rapids.sql.concurrentGpuTasks=1 \
#        --driver-memory 2G \
#        --conf spark.executor.memory=4G \
#        --conf spark.executor.resource.gpu.amount=1 \
#        --conf spark.task.resource.gpu.amount=0.25 \
#        --conf spark.rapids.memory.pinnedPool.size=2G \
#        --conf spark.locality.wait=0s \
#        --conf spark.sql.files.maxPartitionBytes=512m \
#        --conf spark.plugins=com.nvidia.spark.SQLPlugin\
#        --conf spark.rapids.sql.explain="NOT_ON_GPU"\
#        --conf spark.rapids.sql.udfCompiler.enabled="true"
                
# /opt/spark/bin/pyspark \
#        --master spark://hpcuda2:7077 \
#        --conf spark.executor.extraClassPath=/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar\
#        --conf spark.driver.extraClassPath=/opt/sparkRapidsPlugin/cudf-0.18.1-cuda11.jar:/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.4.0.jar \
#        --conf spark.rapids.memory.gpu.allocFraction=0.01 \
#        --conf spark.rapids.sql.concurrentGpuTasks=1 \
#        --driver-memory 2G \
#        --conf spark.executor.memory=4G \
#        --conf spark.executor.resource.gpu.amount=1 \
#        --conf spark.task.resource.gpu.amount=0.25 \
#        --conf spark.rapids.memory.pinnedPool.size=2G \
#        --conf spark.locality.wait=0s \
#        --conf spark.sql.files.maxPartitionBytes=512m \
#        --conf spark.plugins=com.nvidia.spark.SQLPlugin\
#        --conf spark.rapids.sql.explain="NOT_ON_GPU"\
#        --conf spark.rapids.sql.udfCompiler.enabled="true"
                
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_is_fitted
class UserTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.user_map=None
    def _reset(self):
        self.user_map=None
    def fit(self,df):
        self._reset()
        tmp_df=df.select("a_user_id").union(df.select("b_user_id")).withColumnRenamed("a_user_id","user_id_ori")
        self.user_map=tmp_df.distinct().withColumn("user_idx", F.monotonically_increasing_id())
    def transform(self,df):
        df=df.join(self.user_map,df["a_user_id"]==self.user_map["user_id_ori"],how="left").drop("user_id_ori").drop("a_user_id").withColumnRenamed("user_idx","a_user_id")
        df=df.join(self.user_map,df["b_user_id"]==self.user_map["user_id_ori"],how="left").drop("user_id_ori").drop("b_user_id").withColumnRenamed("user_idx","b_user_id")
        return df
    
class PreprocessTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()
        self.a_tweet_id_count=None
        self.b_tweet_id_count=None
        self.a_b_df=None
        self.Vectorizers={}
        self.StringIndexer=None
        self.user_map=None
    def _reset(self):
        self.a_tweet_id_count=None
        self.b_tweet_id_count=None
        self.a_b_df=None
        self.Vectorizers={}
        self.StringIndexer=None
        self.user_map=None
        
    def fit_transform(self, df):
        # Reset internal state before fitting
        self._reset()
        
        arraytimestampCols=["reply","retweet","retweet_comment","like"]
        array_from_unixtime = F.udf(lambda arr: 1 if arr!= None else 0,
                           IntegerType())
        for timestampCol in arraytimestampCols:
            df=df.withColumn(timestampCol,
                array_from_unixtime(df[timestampCol]))
        
        
        self.a_tweet_id_count=df.groupby('a_user_id').agg(F.count("tweet_id").alias("a_tweet_id_count"))
        df=df.join(self.a_tweet_id_count,on="a_user_id",how="left")
        
        self.b_tweet_id_count=df.groupby('b_user_id').agg(F.count("tweet_id").alias("b_tweet_id_count"))
        df=df.join(self.b_tweet_id_count,on="b_user_id",how="left")
​
        self.a_b_df=df.groupby("a_user_id",'b_user_id').agg(F.count(F.lit(1)).alias("a_b_count"))
        df=df.join(self.a_b_df,on=["a_user_id",'b_user_id'])
        
        print(df.select(*target).collect()[0].asDict())
        
        counting_udf = F.udf(lambda x: str(x) if x else 0,
                           IntegerType())
        split_string = F.udf(lambda v:v.split("\t") if v else [],
                           ArrayType(StringType()))
        vector_indices = F.udf(lambda v:v.indices.tolist(),
                           ArrayType(IntegerType()))
        hashCols=["hashtags","links","domains"]
        for hashCol in hashCols:
            df = df.withColumn(hashCol, split_string(hashCol))
            df = df.withColumn(hashCol+"_len", F.size(hashCol))
            self.Vectorizers[hashCol] = CountVectorizer(minDF =10,vocabSize=1000000,inputCol=hashCol, outputCol=hashCol+"_tokens")
            self.Vectorizers[hashCol]=self.Vectorizers[hashCol].fit(df)
            print(len(self.Vectorizers[hashCol].vocabulary))
            df = self.Vectorizers[hashCol].transform(df)
            df = df.withColumn(hashCol,vector_indices(hashCol+"_tokens"))
            df = df.drop(hashCol+"_tokens")
​
        #hashing
        labelize_inputCols=["media","tweet_type","language"]
        labelize_outputCols=[inputCol+"_transformed" for inputCol in labelize_inputCols]
        self.StringIndexer = StringIndexer(inputCols=labelize_inputCols, outputCols=labelize_outputCols,handleInvalid="keep")
        self.StringIndexer = self.StringIndexer.fit(df)
        df =self.StringIndexer.transform(df)
        for labelize_inputCol in labelize_inputCols:
            df = df.drop(labelize_inputCol).withColumnRenamed(labelize_inputCol+"_transformed", labelize_inputCol)
        return df
    def transform(self, df, update=True):
        arraytimestampCols=["reply","retweet","retweet_comment","like"]
        array_from_unixtime = F.udf(lambda arr: 1 if arr!= None else 0,
                           IntegerType())
        for timestampCol in arraytimestampCols:
            df=df.withColumn(timestampCol,
                array_from_unixtime(df[timestampCol]))
            
        if update:
            update_tweet_id_count=df.groupby('a_user_id').agg(F.count("tweet_id").alias("a_tweet_id_count_update"))
            self.a_tweet_id_count=self.a_tweet_id_count.join(update_tweet_id_count,on="a_user_id",how="outer").withColumn("a_tweet_id_count",F.col("a_tweet_id_count")+F.col("a_tweet_id_count_update")).drop("a_tweet_id_count_update")
        df=df.join(self.a_tweet_id_count,on="a_user_id",how="left")
        
        if update:
            update_tweet_id_count=df.groupby('b_user_id').agg(F.count("tweet_id").alias("b_tweet_id_count_update"))
            self.b_tweet_id_count=self.b_tweet_id_count.join(update_tweet_id_count,on="b_user_id",how="outer").withColumn("b_tweet_id_count",F.col("b_tweet_id_count")+F.col("b_tweet_id_count_update")).drop("b_tweet_id_count_update")
        df=df.join(self.b_tweet_id_count,on="b_user_id",how="left")
​
        if update:
            update_a_b_df=df.groupby("a_user_id",'b_user_id').agg(F.count(F.lit(1)).alias("a_b_count_update"))
            self.a_b_df=self.a_b_df.join(update_a_b_df,on=["a_user_id",'b_user_id'],how="outer").withColumn("a_b_count",F.col("a_b_count")+F.col("a_b_count_update")).drop("a_b_count_update")
        df=df.join(self.a_b_df,on=["a_user_id",'b_user_id'])
        
        print(df.select(*target).collect()[0].asDict())
        hashCols=["hashtags","links","domains"]
        counting_udf = F.udf(lambda x: str(x) if x else 0,
                           IntegerType())
        split_string = F.udf(lambda v:v.split("\t") if v else [],
                           ArrayType(StringType()))
        vector_indices = F.udf(lambda v:v.indices.tolist(),
                           ArrayType(IntegerType()))
        for hashCol in hashCols:
            df = df.withColumn(hashCol, split_string(hashCol))
            df = df.withColumn(hashCol+"_len", F.size(hashCol))
            print(len(self.Vectorizers[hashCol].vocabulary))
            df = self.Vectorizers[hashCol].transform(df)
            df = df.withColumn(hashCol,vector_indices(hashCol+"_tokens"))
            df = df.drop(hashCol+"_tokens")
​
        #hashing
        labelize_inputCols=["media","tweet_type","language"]
        labelize_outputCols=[inputCol+"_transformed" for inputCol in labelize_inputCols]
        df =self.StringIndexer.transform(df)
        for labelize_inputCol in labelize_inputCols:
            df = df.drop(labelize_inputCol).withColumnRenamed(labelize_inputCol+"_transformed", labelize_inputCol)
        return df
​
filenames = [
    f'{base_dir}/training/pivoted_tsv/part-00{i:03d}'.format(i=i)
    for i in range(files_nums)
]
​
​
​
valid_filenames = [
    f'{base_dir}/training/pivoted_tsv/part-00{i:03d}'.format(i=i)
    for i in range(files_nums,files_nums+files_nums)
]
​
# file_list=glob.glob(f'{base_dir}/training/pivoted_parquet/*parquet')
# filenames = file_list[:50]
​
​
​
# valid_filenames = file_list[50:50+50]
​
​
# filenames = [
#     '/tmp2/recsys2021/training.tsv'
# ]
columns = [
    # features
    'text_tokens', 'hashtags', 'tweet_id',
    'media', 'links', 'domains', 'tweet_type', 'language', 'timestamp',
    'a_user_id', 'a_follower_count', 'a_following_count', 'a_is_verified', 'a_account_creation',
    'b_user_id', 'b_follower_count', 'b_following_count', 'b_is_verified', 'b_account_creation',
    'a_follows_b',
    # labels
    'reply', 'retweet', 'retweet_comment', 'like',
]
target_col=['reply', 'retweet', 'retweet_comment', 'like']
target=[F.mean(label).alias("mean_"+label) for label in target_col]
target_count=[F.sum(label).alias("count_"+label) for label in target_col]
array_cols=["hashtags","links","domains"]
schema = StructType([
    StructField("text_tokens", StringType(), True),
    StructField("hashtags", StringType(), True),
    StructField("tweet_id", StringType(), True),
    StructField("media", StringType(), True),
    StructField("links", StringType(), True),
    StructField("domains", StringType(), True),
    StructField("tweet_type", StringType(), True),
    StructField("language", StringType(), True),
    StructField("timestamp", IntegerType(), True), 
    StructField("a_user_id", StringType(), True),
    StructField("a_follower_count", IntegerType(), True),
    StructField("a_following_count", IntegerType(), True),
    StructField("a_is_verified", BooleanType(), True),
    StructField("a_account_creation", IntegerType(), True),
    StructField("b_user_id", StringType(), True),
    StructField("b_follower_count", IntegerType(), True),
    StructField("b_following_count", IntegerType(), True),
    StructField("b_is_verified", BooleanType(), True),
    StructField("b_account_creation", IntegerType(), True),
    StructField("a_follows_b", BooleanType(), True),
    StructField("reply", StringType(), True),
    StructField("retweet", StringType(), True),
    StructField("retweet_comment", StringType(), True),
    StructField("like", StringType(), True)
    ])
# df=spark.read.csv(filenames, encoding='utf-8', sep='\x01',header=False,schema=schema)
# df.write.parquet(f'{base_dir}/training/pivoted_parquet/',mode="overwrite")
# 1/0
train_df=spark.read.csv(filenames, encoding='utf-8', sep='\x01',header=False,schema=schema).cache()
valid_df=spark.read.csv(valid_filenames, encoding='utf-8', sep='\x01',header=False,schema=schema).cache()
​
# train_df=spark.read.parquet(filenames).persist(storagelevel.StorageLevel.MEMORY_ONLY).coalesce(files_nums//10*concurrentGpuTasks*gpu_amount)
# valid_df=spark.read.parquet(valid_filenames).persist(storagelevel.StorageLevel.MEMORY_ONLY).coalesce(files_nums//10*concurrentGpuTasks*gpu_amount)
class TETransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.user_map=None
    def _reset(self):
        self.user_map=None
    def fit_transform(self,train_df,valid_df,seed,cols,smooth,labels):
        #get valid_df TE
        target=[F.mean(label).alias("mean_"+label) for label in labels]
        te_colname='_'.join(cols)
        te_df=train_df.groupby(cols).agg(F.count(F.lit(1)).alias("count"),*target)
        global_means=train_df.select(*target).collect()[0].asDict()
        for k,global_mean in global_means.items():
            label=k[4:]
            te_df = te_df.withColumn("TE"+te_colname+label,((te_df['count']*te_df['mean'+label])+(global_mean*smooth)) / (te_df['count']+smooth))
        valid_df=valid_df.join(other=te_df.select(cols+["TE"+te_colname+'_'+label for label in labels ]+["count"]),on=cols,how="left")
​
        #get train K-fold TE
        results=[]
        kfolds=df.randomSplit([0.2 for i in range(5)],seed)
        for i in range(5):
            others_df=[kfolds[j] for j in range(5) if j!=i]
            others_df = reduce(DataFrame.unionByName, others_df)
            te_df=others_df.groupby(cols).agg(F.count(F.lit(1)).alias("count"),*target)
            global_means=others_df.select(*target).collect()[0].asDict()
            for k,global_mean in global_means.items():
                label=k[4:]
                te_df = te_df.withColumn("TE"+te_colname+label,((te_df['count']*te_df['mean'+label])+(global_mean*smooth)) / (te_df['count']+smooth))
            results.append(kfolds[i].join(other=te_df.select(cols+["TE"+te_colname+'_'+label for label in labels]+["count"]),on=cols,how="left"))
        train_df = reduce(DataFrame.unionByName, results)
        print(te_colname)
        return train_df,valid_df
class TE_arrayTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.user_map=None
    def _reset(self):
        self.user_map=None
    def fit_transform(self,train_df,valid_df,seed,cols,smooth,labels):
        #get valid_df TE
        target=[F.mean(label).alias("mean_"+label) for label in labels]
        te_colname='_'+cols
        te_df=train_df.withColumn(cols,F.explode(cols)).groupby(cols).agg(F.count(F.lit(1)).alias("count"),*target)
        global_means=train_df.select(*target).collect()[0].asDict()
        for k,global_mean in global_means.items():
            label=k[4:]
            te_df = te_df.withColumn("TE"+te_colname+label,((te_df['count']*te_df['mean'+label])+(global_mean*smooth)) / (te_df['count']+smooth))
        mean_valid_df=valid_df.withColumn(cols,F.explode(cols)).join(other=te_df.select([cols]+["TE"+te_colname+'_'+label for label in labels ]+["count"]),on=cols,how="left").\
        groupby(["a_user_id","b_user_id","timestamp",'tweet_id']).agg(F.count(F.lit(1)).alias("count"),*[F.mean("TE"+te_colname+'_'+label).alias("TE"+te_colname+'_'+label) for label in labels ])
        print("before array:",valid_df.count(),mean_valid_df.count())
        valid_df=valid_df.join(other=mean_valid_df.select(["a_user_id","b_user_id","timestamp",'tweet_id']+["TE"+te_colname+'_'+label for label in labels ]+["count"]),on=["a_user_id","b_user_id","timestamp",'tweet_id'],how="left")
        print("after array:",valid_df.count())
        #get train K-fold TE
        results=[]
        kfolds=train_df.randomSplit([0.2 for i in range(5)],seed)
        for i in range(5):
            others_df=[kfolds[j] for j in range(5) if j!=i]
            others_df = reduce(DataFrame.unionByName, others_df)
            te_df=others_df.withColumn(cols,F.explode(cols)).groupby(cols).agg(F.count(F.lit(1)).alias("count"),*target)
            global_means=others_df.select(*target).collect()[0].asDict()
            for k,global_mean in global_means.items():
                label=k[4:]
                te_df = te_df.withColumn("TE"+te_colname+label,((te_df['count']*te_df['mean'+label])+(global_mean*smooth)) / (te_df['count']+smooth))
            mean_kfolds=kfolds[i].withColumn(cols,F.explode(cols)).join(other=te_df.select([cols]+["TE"+te_colname+'_'+label for label in labels ]+["count"]),on=cols,how="left").\
            groupby(["a_user_id","b_user_id","timestamp",'tweet_id']).agg(F.count(F.lit(1)).alias("count"),*[F.mean("TE"+te_colname+'_'+label).alias("TE"+te_colname+'_'+label) for label in labels ])
            print("before array:",kfolds[i].count())
            result=kfolds[i].join(other=mean_kfolds.select(["a_user_id","b_user_id","timestamp",'tweet_id']+["TE"+te_colname+'_'+label for label in labels ]+["count"]),on=["a_user_id","b_user_id","timestamp",'tweet_id'],how="left")
            print("after array:",mean_kfolds)
            print("after array:",result.count())
            results.append(result)
        train_df = reduce(DataFrame.unionByName, results)
        print(te_colname)
        return train_df,valid_df
seed=42
cols=["tweet_type","language"]
smooth=20
labels=['reply', 'retweet', 'retweet_comment', 'like']
# for c in [
#     ['b_user_id','tweet_type','language'],
#     ['a_user_id'],
#     ['b_user_id'],
#     ['a_user_id'],
#     ['b_user_id','b_user_id'],
​
# ]:
​
preprocessTransformer=PreprocessTransformer()
train_df=preprocessTransformer.fit_transform(train_df)
valid_df=preprocessTransformer.transform(valid_df,update=True)
userTransformer=UserTransformer()
userTransformer.fit(train_df)
train_df=userTransformer.transform(train_df)
valid_df=userTransformer.transform(valid_df)
​
​
{'mean_reply': 0.029647463306004047, 'mean_retweet': 0.08759442850616271, 'mean_retweet_comment': 0.0070908526189384655, 'mean_like': 0.3980828889226523}
429084
243842
66057
{'mean_reply': 0.029662613031652556, 'mean_retweet': 0.08759684056790065, 'mean_retweet_comment': 0.00709082605072992, 'mean_like': 0.3980664735411461}
429084
243842
66057
features_col = [feat for feat in train_df.schema.names if feat not in target_col ]
params = {'eta': 0.1, 'gamma': 0.1, 'missing': 0.0,
          'treeMethod': 'gpu_hist', 'maxDepth': 3, 
          'growPolicy': 'depthwise', 'lambda_': 1.0,
          'subsample': 1.0, 'numRound': 1000,
          'numWorkers': 1, 'verbosity': 1}
tE_arrayTransformer=TE_arrayTransformer()
seed=42
cols=["tweet_type","language"]
smooth=20
labels=['reply', 'retweet', 'retweet_comment', 'like']
for array_col in array_cols:
    train_df,valid_df=tE_arrayTransformer.fit_transform(train_df,valid_df,seed,array_col,smooth,labels)

xgboost = XGBoostClassifier(**params).setLabelCol(target_col).setFeaturesCols(features_col)
model = xgboost.fit(train_df)
features_col = [feat for feat in train_df.schema.names if feat not in target_col ]
params = {'eta': 0.1, 'gamma': 0.1, 'missing': 0.0,
          'treeMethod': 'gpu_hist', 'maxDepth': 3, 
          'growPolicy': 'depthwise', 'lambda_': 1.0,
          'subsample': 1.0, 'numRound': 1000,
          'numWorkers': 1, 'verbosity': 1}
tE_arrayTransformer=TE_arrayTransformer()
seed=42
cols=["tweet_type","language"]
smooth=20
labels=['reply', 'retweet', 'retweet_comment', 'like']
for array_col in array_cols:
    train_df,valid_df=tE_arrayTransformer.fit_transform(train_df,valid_df,seed,array_col,smooth,labels)
​
xgboost = XGBoostClassifier(**params).setLabelCol(target_col).setFeaturesCols(features_col)
model = xgboost.fit(train_df)

