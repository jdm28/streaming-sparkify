import os
import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    song_data = 'song_data/*/*/*/*.json'
    
    df = spark.read.json(song_data)
    
    songs_table =  df.select('song_id', 'title', 'artist_id', 'year', 'duration')
    songs_table.write.partitionBy("year", "artist_id").parquet(output_data + "songs.parquet")
    
    artists_table = df.select('artist_id','artist_name','artist_location','artist_latitude','artist_longitude')
    artists_table.write.parquet(output_data + "artists_table.parquet")


def process_log_data(spark, input_data, output_data):
    log_data = 'log_data/*/*/*.json'

    df = spark.read.json(log_data)
    df = df.filter(df['page'] == 'NextSong')

    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level')
    users_table.write.parquet(output_data + "users_table.parquet")

    get_timestamp = udf(lambda x : datetime.utcfromtimestamp(int(x)/1000), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    get_datetime = udf(lambda x : datetime.fromtimestamp(x/ 1000.0).strftime("%Y-%m-%d %H:%M:%S"))
    df = df.withColumn("datetime", get_timestamp(df.ts)) 
    
    time_table = df.withColumn("hour",hour(df.datetime))\
                .withColumn("day",dayofmonth(df.datetime)) \
                .withColumn("week",weekofyear(df.datetime)) \
                .withColumn("month",month(df.datetime)) \
                .withColumn("year",year(df.datetime)) \
                .withColumn("weekday",date_format(df.datetime, 'u')) \
                .selectExpr('ts AS start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday').dropDuplicates() 
    
    time_table.write.partitionBy("year", "month").parquet(output_data + "time_table.parquet")

    song_df = spark.read.format('parquet').load(output_data + 'songs.parquet') 

    songplays_table = songs_plays = df.join(songs, df.song == songs.title, how='inner').withColumnRenamed("year","yearSongs")\
                 .join(time_table, time_table.start_time == df.ts, how='inner')\
                .selectExpr("monotonically_increasing_id() AS songplay_id"
                                   ,"ts as start_time"
                                   ,"userId AS user_id"
                                   ,"level"
                                   ,"level","song_id","artist_id"
                                   ,"sessionId AS session_id"
                                   ,"location"
                                   ,"userAgent AS user_agent"
                                   ,"month"
                                   ,"year"
                                   )
    songplays_table.write.partitionBy("year", "month").parquet(output_data + "time_table.parquet")


def main():
    spark = create_spark_session()
    input_data = config['INPUT_DATA']
    output_data = config['OUTPUT_DATA']
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
