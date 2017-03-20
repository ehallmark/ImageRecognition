package main.java.flicker_scraper;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/18/2017.
 */
public class ReadAndSaveFileListFromGCS {
    public static String IMAGE_LOCATIONS_FILE = "/home/ehallmark1122/ImageRecognition/actual_image_locations.txt";
    public static void main(String[] args) throws Exception {
        boolean useSparkLocal = true;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        AtomicInteger cnt = new AtomicInteger(0);
        sparkConf.setAppName("ReadAndSaveFileListFromGCS");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        List<String> data = sc.wholeTextFiles("gs://image-scrape-dump/images",50).map(line->{
            System.out.println("Line: "+line._1);
            System.out.println("Count: " + cnt.getAndIncrement());
            return line._1;
            /*try {
                final URL url = new URL("https://storage.googleapis.com/image-scrape-dump/images/" + line.hashCode() + ".jpg");
                if (ImageStreamer.loadImage(url)!=null) {
                    System.out.println("GOOD");
                    return line;
                } else {
                    System.out.println("BAD");
                }
            } catch( Exception e) {
                e.printStackTrace();
            }
            return null;*/

        }).collect();
        System.out.println("Started saving");
        BufferedWriter writer = new BufferedWriter(new FileWriter(IMAGE_LOCATIONS_FILE));
        data.forEach(url->{
            try {
                writer.write(url+"\n");
            }catch(Exception e) {
                e.printStackTrace();
            }
        });
        writer.flush();
        writer.close();
        System.out.println("Finished");
    }
}
