package main.java.flicker_scraper;

import main.java.data_loader.DataLoader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.TypedColumn;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
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

        MnistDataSetIterator train = new MnistDataSetIterator(1,60000,true,true,true,seed);
        MnistDataSetIterator test = new MnistDataSetIterator(1,10000,true,false,true,seed);
        saveToBucket(train,"mnist-train",spark);
        saveToBucket(test,"mnist-test",spark);
    }

    private static void saveToBucket(MnistDataSetIterator iter, String bucketName, SparkSession spark) {
        List<FeatureLabelPair> data = new ArrayList<>();
        while(iter.hasNext()) {
            DataSet dataSet = iter.next();
            if(dataSet==null||dataSet.getFeatureMatrix()==null||dataSet.getLabels()==null) continue;
            FeatureLabelPair pair = new FeatureLabelPair();
            pair.setFeatures(dataSet.getFeatureMatrix().data().asFloat());
            pair.setLabels(dataSet.getLabels().data().asFloat());
            data.add(pair);
        }
        System.out.println("Info for: "+bucketName);
        System.out.println("Num datasets: "+data.size());
        spark.createDataset(data, Encoders.bean(FeatureLabelPair.class))
                .write()
                .format(FlickrScraper.AVRO_FORMAT)
                .save(FlickrScraper.LABELED_IMAGES_BUCKET+bucketName);
    }

    private static JavaRDD<DataSet> getData(String bucket, SparkSession spark, int batch, int nInputs, int nOutputs) {
        return DataLoader.batchBy(spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(FlickrScraper.LABELED_IMAGES_BUCKET+bucket)
                .as(Encoders.bean(FeatureLabelPair.class))
                .toJavaRDD().repartition(100)
                .map(pair->{
                    return new DataSet(Nd4j.create(pair.getFeatures()), Nd4j.create(pair.getLabels()));
                }),batch,nInputs,nOutputs);
    }

    private static JavaRDD<DataSet> getAutoEncoderData(String bucket, SparkSession spark, int batch, int nInputs) {
        return DataLoader.batchBy(spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(FlickrScraper.LABELED_IMAGES_BUCKET+bucket)
                .as(Encoders.bean(FeatureLabelPair.class))
                .toJavaRDD().repartition(100)
                .map(pair->{
                    return new DataSet(Nd4j.create(pair.getFeatures()), Nd4j.create(pair.getFeatures()));
                }),batch,nInputs,nInputs);    }

    public static JavaRDD<DataSet> getTrainData(SparkSession spark, int batch, int nInputs, int nOutputs) {
        return getData("mnist-train",spark,batch,nInputs,nOutputs);
    }

    public static JavaRDD<DataSet> getTestData(SparkSession spark, int batch, int nInputs, int nOutputs) {
        return getData("mnist-test",spark,batch,nInputs,nOutputs);
    }

    public static JavaRDD<DataSet> getTrainAutoEncoderData(SparkSession spark, int batch, int nInputs) {
        return getAutoEncoderData("mnist-train",spark,batch,nInputs);
    }

    public static JavaRDD<DataSet> getTestAutoEncoderData(SparkSession spark, int batch, int nInputs) {
        return getAutoEncoderData("mnist-test",spark,batch,nInputs);
    }

}
