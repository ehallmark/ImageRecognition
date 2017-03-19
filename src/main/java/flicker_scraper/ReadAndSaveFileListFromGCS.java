package main.java.flicker_scraper;

import main.java.SparkAutoEncoder;
import main.java.image_vectorization.ImageStreamer;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/18/2017.
 */
public class ReadAndSaveFileListFromGCS {
    public static String IMAGE_LOCATIONS_FILE = "/mnt/bucket/actual_image_locations.txt";
    public static void main(String[] args) throws Exception {
        boolean useSparkLocal = true;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        AtomicInteger cnt = new AtomicInteger(0);
        sparkConf.setAppName("ReadAndSaveFileListFromGCS");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        sc.textFile("gs://image-scrape-dump/all_flickr_urls.txt").map(line->{
            System.out.println("Line: "+line);
            System.out.println("Count: " + cnt.getAndIncrement());
            try {
                return sc.binaryFiles("gs://image-scrape-dump/images/" + line.hashCode() + ".jpg").count() > 0;
            } catch( Exception e) {
                e.printStackTrace();
                return null;
            }
        }).filter(x->x!=null).saveAsTextFile("gs://image-scrape-dump/actual_image_locations.txt");
        System.out.println("Finished");
    }
}
