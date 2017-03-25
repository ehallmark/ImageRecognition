package main.java.data_loader;

import main.java.flicker_scraper.FlickrScraper;
import main.java.flicker_scraper.Image;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
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

    public static Dataset<Row> loadDataNames(SparkSession spark, String bucket) {
        return spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(bucket);
    }

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
                .map(row->{
                    if(row.isNullAt(0) || row.isNullAt(1)) {
                        System.out.println("Row has a null!");
                        return null;
                    }
                    Image image = new Image();
                    image.setImage((byte[])row.get(0));
                    image.setCategory((String)row.get(1));
                    return image;
                },Encoders.bean(Image.class)).filter(image->image!=null).toJavaRDD()
                .map(image-> {
                    INDArray vec;
                    try {
                        String label = classifyFolderNames ? null : image.getCategory();
                        Integer idx = invertedIdxMap.get(label);
                        if (idx == null || idx < 0) {
                            System.out.println("Invalid label: " + label);
                            return null;
                        } else {
                            INDArray labelVec = Nd4j.zeros(numOutputs);
                            BufferedImage jpg = ImageIO.read(new ByteArrayInputStream(image.getImage()));
                            if (jpg.getHeight() != height || jpg.getWidth() != width) {
                                jpg = Scalr.resize(jpg, Scalr.Method.ULTRA_QUALITY, height, width, Scalr.OP_ANTIALIAS);
                            }
                            if (jpg == null) return null;
                            vec = ImageVectorizer.vectorizeImage(jpg, numInputs);
                            if (vec != null) {
                                labelVec.putScalar(idx, 1.0);
                                return new DataSet(vec, labelVec);
                            }
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return null;

                }).filter(d->d!=null);

        return data;
    }

    public static JavaRDD<DataSet> loadAutoEncoderData(SparkSession spark, int height, int width, int channels, String... bucketNames) {
        int numInputs = height*width*channels;
        JavaRDD<DataSet> data = spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(bucketNames)
                .select("image")
                .javaRDD()
                .map((Row row) -> {
                    INDArray vec;
                    try {
                        if(row.isNullAt(0)) {
                            System.out.println("Row has a null!");
                            return null;
                        }
                        BufferedImage image = ImageIO.read(new ByteArrayInputStream((byte[]) (row.get(0))));
                        if(image.getHeight()!=height||image.getWidth()!=width) {
                            image = Scalr.resize(image,Scalr.Method.ULTRA_QUALITY,height,width,Scalr.OP_ANTIALIAS);
                        }
                        vec = ImageVectorizer.vectorizeImage(image, numInputs);
                        if(vec!=null) {
                            return new DataSet(vec, vec);
                        }

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return null;

                }).filter(d->d!=null);

        return data;
    }
}
