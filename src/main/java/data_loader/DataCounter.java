package main.java.data_loader;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Arrays;

/**
 * Created by Evan on 3/25/2017.
 */
public class DataCounter {
    public static void main(String[] args) throws Exception {
        // Spark stuff
        boolean useSparkLocal = false;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("Image Recognition Data Counter");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        SparkSession spark = SparkSession.builder()
                .appName("FlickrScraper")
                .getOrCreate();

        if(args.length<1) {
            throw new RuntimeException("Must specify databucket argument");
        }

        String dataBucketNamePrefix = "gs://image-scrape-dump/labeled-images/";
        Arrays.stream(args).forEach(arg->{
            Dataset<Row> data = DataLoader.loadDataNames(spark,dataBucketNamePrefix+arg);
            System.out.println("Count for "+arg+" Dataset: "+data.count());
        });
    }
}
