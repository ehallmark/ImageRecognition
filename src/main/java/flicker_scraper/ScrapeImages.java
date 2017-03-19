package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/15/2017.
 */
public class ScrapeImages {
    public static final String IMAGE_DIR = "/mnt/bucket/images/";
    public static void main(String[] args) throws Exception{
        boolean useSparkLocal = false;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("ScrapeImages");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> lines = sc.textFile(MergeUrlFiles.mergedFile.getAbsolutePath());

        AtomicInteger cnt = new AtomicInteger(0);
        lines.foreach(line->{
            System.out.print(cnt.getAndIncrement()+" - ");
            if(trySaveImageToGoogleCloud(line)) {
                System.out.print("found");
            }
            System.out.println();
        });
    }

    public static boolean trySaveImageToGoogleCloud(String urlString) {
        try {
            File file = new File(IMAGE_DIR+urlString.hashCode()+".jpg");
            if(!file.exists()) {
                BufferedImage image = ImageStreamer.loadImage(new URL(urlString));
                if(image!=null) {
                    ImageIO.write(image, "jpg", file);
                    return true;
                }
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
        return false;
    }
}
