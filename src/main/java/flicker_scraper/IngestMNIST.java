package main.java.flicker_scraper;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.TypedColumn;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Evan on 4/2/2017.
 */
public class IngestMNIST {
    public static void main(String[] args) throws Exception {
        final int seed = 69;
        boolean useSparkLocal = false;
        //int numPartitions = 150;

        SparkConf sparkConf = new SparkConf();

        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("FlickrScraper");
        SparkSession spark = SparkSession.builder()
                .appName("FlickrScraper")
                .config(sparkConf)
                .getOrCreate();

        MnistDataSetIterator train = new MnistDataSetIterator(50,60000,false,true,true,seed);
        MnistDataSetIterator test = new MnistDataSetIterator(50,10000,false,false,true,seed);
        saveToBucket(train,"mnist-train",spark);
        saveToBucket(test,"mnist-test",spark);
    }

    private static void saveToBucket(MnistDataSetIterator iter, String bucketName, SparkSession spark) {
        List<FeatureLabelPair> data = new ArrayList<>();
        while(iter.hasNext()) {
            DataSet dataSet = iter.next();
            if(dataSet==null) continue;
            data.add(new FeatureLabelPair(dataSet.getFeatureMatrix(),dataSet.getLabels()));
        }
        System.out.println("Info for: "+bucketName);
        System.out.println("Num datasets: "+data.size());
        spark.createDataset(data, Encoders.bean(FeatureLabelPair.class))
                .write()
                .format(FlickrScraper.AVRO_FORMAT)
                .mode(SaveMode.Overwrite)
                .save(FlickrScraper.LABELED_IMAGES_BUCKET+bucketName);
    }

    private static JavaRDD<DataSet> getData(String bucket, SparkSession spark) {
        return spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(FlickrScraper.LABELED_IMAGES_BUCKET+bucket)
                .as(Encoders.bean(FeatureLabelPair.class))
                .toJavaRDD()
                .map(pair->{
                    return new DataSet(pair.getFeatures(),pair.getLabels());
                });
    }

    public static JavaRDD<DataSet> getTrainData(SparkSession spark) {
        return getData("mnist-train",spark);
    }

    public static JavaRDD<DataSet> getTestData(SparkSession spark) {
        return getData("mnist-test",spark);
    }
}
