package main.java.image_vectorization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/11/2017.
 */
public class ImageVectorizer {

    public static INDArray vectorizeImage(final BufferedImage image, int numInputs, boolean blackAndWhite) {
        // Getting pixel color by position x and y
        int startX = 0;
        int startY = 0;
        int endX = image.getWidth();
        int endY = image.getHeight();
        if(image.getHeight()>endY||startY<0||startY>=endY||image.getWidth()>endX||startX<0||startX>=endX) {
            throw new RuntimeException("invalid dimensions");
        }
        INDArray vec = Nd4j.zeros(numInputs);
        AtomicInteger idx = new AtomicInteger(0);
        for(int x = startX; x < endX; x++) {
            for(int y = startY; y < endY; y++) {
                Color color = new Color(image.getRGB(x,y),true);
                int  red   = color.getRed();
                int  green = color.getGreen();
                int  blue  =  color.getBlue();
                double black = new Double(red+green+blue)/255d;
                if(blackAndWhite) {
                    vec.putScalar(idx.getAndIncrement(),black);
                } else {
                    for (int c : new int[]{red, green, blue}) {
                        vec.putScalar(idx.getAndIncrement(), new Double(c) / 255);
                        if (idx.get() >= numInputs) return vec;
                    }
                }
            }
        }
        return vec;
    }
}
