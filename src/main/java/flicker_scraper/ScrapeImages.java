package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/15/2017.
 */
public class ScrapeImages {
    public static final String IMAGE_DIR = "/mnt/bucket/images/";
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(MergeUrlFiles.mergedFile));
        AtomicInteger cnt = new AtomicInteger(0);
        ForkJoinPool pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors()*4);
        reader.lines().forEach(line->{
            pool.execute(new RecursiveAction() {
                @Override
                protected void compute() {
                    if(trySaveImageToGoogleCloud(line)) {
                        System.out.println(cnt.getAndIncrement());
                    }
                }
            });
        });
        pool.shutdown();
        pool.awaitTermination(Long.MAX_VALUE, TimeUnit.MICROSECONDS);
    }

    public static boolean trySaveImageToGoogleCloud(String urlString) {
        try {
            BufferedImage image = ImageStreamer.loadImage(new URL(urlString));
            if(image!=null) {
                File file = new File(IMAGE_DIR+urlString.hashCode()+".jpg");
                if(!file.exists()) {
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
