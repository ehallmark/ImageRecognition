package main.java.data_loader;

import main.java.flicker_scraper.FlickrScraper;
import main.java.flicker_scraper.Image;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.imgscalr.Scalr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Evan on 3/21/2017.
 */
public class DataLoader {
    public static JavaRDD<DataSet> loadClassificationData(SparkSession spark, int height, int width, int channels, List<String> labels, boolean classifyFolderNames, String... bucketNames) {
        Map<String,Integer> invertedIdxMap = new HashMap<>();
        for(int i = 0; i < labels.size(); i++) {
            invertedIdxMap.put(labels.get(i),i);
        }
        int numInputs = height*width*channels;
        int numOutputs = labels.size();
        JavaRDD<DataSet> data = spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(bucketNames)
                .select("image","category")
                .javaRDD()
                .map((Row row) -> {
                    INDArray vec;
                    try {
                        if(row.isNullAt(0) || row.isNullAt(1)) {
                            System.out.println("Row has a null!");
                            return null;
                        }
                        String label = classifyFolderNames ? row.getrow.get(1).toString();
                        Integer idx = invertedIdxMap.get(label);
                        if(idx==null || idx<0) {
                            System.out.println("Invalid label: "+label);
                            return null;
                        } else {
                            INDArray labelVec = Nd4j.zeros(numOutputs);
                            BufferedImage image = ImageIO.read(new ByteArrayInputStream((byte[]) (row.get(0))));
                            if(image.getHeight()!=height||image.getWidth()!=width) {
                                image = Scalr.resize(image,Scalr.Method.ULTRA_QUALITY,height,width,Scalr.OP_ANTIALIAS);
                            }
                            vec = ImageVectorizer.vectorizeImage(image, numInputs);
                            labelVec.putScalar(idx, 1.0);
                            return new DataSet(vec, labelVec);
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return null;

                }).filter(d->d!=null);

        return data;
    }
}
