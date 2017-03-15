package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/15/2017.
 */
public class ScrapeImages {
    public static final String IMAGE_DIR = "/mnt/bucket/images/";
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(MergeUrlFiles.mergedFile));
        AtomicInteger cnt = new AtomicInteger(0);
        reader.lines().forEach(line->{
            System.out.println(cnt.getAndIncrement());
            trySaveImageToGoogleCloud(line);
        });
    }

    public static void trySaveImageToGoogleCloud(String urlString) {
        try {
            BufferedImage image = ImageStreamer.loadImage(new URL(urlString));
            if(image!=null) {
                File file = new File(IMAGE_DIR+urlString.hashCode()+".jpg");
                ImageIO.write(image,"jpg",file);
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
