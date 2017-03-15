package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;

/**
 * Created by Evan on 3/15/2017.
 */
public class ScrapeImages {
    public static final String IMAGE_DIR = "flickr_images/";
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(MergeUrlFiles.mergedFile));
        reader.lines().forEach(line->{
            try {
                BufferedImage image = ImageStreamer.loadImage(new URL(line));
                if(image!=null) {
                    File file = new File(IMAGE_DIR+line.hashCode()+".jpg");
                    ImageIO.write(image,"jpg",file);
                }
            } catch(Exception e) {
                e.printStackTrace();
            }
        });
    }
}
