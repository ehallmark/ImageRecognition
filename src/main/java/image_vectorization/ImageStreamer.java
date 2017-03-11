package main.java.image_vectorization;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

/**
 * Created by Evan on 3/11/2017.
 */
public class ImageStreamer {
    public static BufferedImage loadImage(URL url) {
        try {
            BufferedImage image = ImageIO.read(url);
            return image;
        } catch(Exception ioe) {
            System.out.println("Unable to find: "+url.toString());
            return null;
        }
    }
}
