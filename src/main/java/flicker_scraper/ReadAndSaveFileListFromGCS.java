package main.java.flicker_scraper;

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
    public static File IMAGE_LOCATIONS_FILE = new File("/mnt/bucket/actual_image_locations.txt");
    public static void main(String[] args) throws Exception {
        File urls = MergeUrlFiles.mergedFile;
        boolean useSparkLocal = true;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("ReadAndSaveFileListFromGCS");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        AtomicInteger cnt = new AtomicInteger(0);
        BufferedWriter writer = new BufferedWriter(new FileWriter(IMAGE_LOCATIONS_FILE));
        sc.textFile(urls.getAbsolutePath()).foreach(line->{
            int hash = line.hashCode();
            String url = ScrapeImages.IMAGE_DIR+String.valueOf(hash)+".jpg";
            if(new File(url).exists()) {
                try {
                    synchronized (writer){
                        writer.write(url + "\n");
                    }
                    System.out.println(cnt.getAndIncrement());
                } catch(Exception e) {
                    e.printStackTrace();
                }
            }
        });
        System.out.println("Finished");
        writer.flush();
        writer.close();
    }
}
