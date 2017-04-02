package main.java.data_loader;

import main.java.flicker_scraper.FlickrScraper;
import main.java.flicker_scraper.Image;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.imgscalr.Scalr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/21/2017.
 */
public class DataLoader {

    public static Dataset<Row> loadDataNames(SparkSession spark, String bucket) {
        return spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(bucket);
    }

    public static JavaRDD<DataSet> loadClassificationData(SparkSession spark, int height, int width, int channels, List<String> labels, boolean classifyFolderNames, int batchSize, String... bucketNames) {
        Map<String,Integer> invertedIdxMap = new HashMap<>();
        for(int i = 0; i < labels.size(); i++) {
            invertedIdxMap.put(labels.get(i),i);
        }
        int numInputs = height*width*channels;
        int numOutputs = labels.size();
        List<JavaRDD<DataSet>> dataList = new ArrayList<>(bucketNames.length);
        Arrays.stream(bucketNames).forEach(filename-> {
                    JavaRDD<DataSet> data = spark.read()
                            .format(FlickrScraper.AVRO_FORMAT)
                            .load(filename)
                            .select("image", "category")
                            .map(row -> {
                                if (row.isNullAt(0) || row.isNullAt(1)) {
                                    System.out.println("Row has a null!");
                                    return null;
                                }
                                Image image = new Image();
                                image.setImage((byte[]) row.get(0));
                                image.setCategory((String) row.get(1));
                                return image;
                            }, Encoders.bean(Image.class)).filter(image -> image != null).toJavaRDD()
                            .map(image -> {
                                INDArray vec;
                                try {
                                    String label = classifyFolderNames ? filename : image.getCategory();
                                    Integer idx = invertedIdxMap.get(label);
                                    if (idx == null || idx < 0) {
                                        System.out.println("Invalid label: " + label);
                                        return null;
                                    } else {
                                        INDArray labelVec = Nd4j.zeros(numOutputs);
                                        BufferedImage jpg = ImageIO.read(new ByteArrayInputStream(image.getImage()));
                                        if (jpg.getHeight() != height || jpg.getWidth() != width) {
                                            jpg = Scalr.resize(jpg, Scalr.Method.QUALITY, height, width, Scalr.OP_ANTIALIAS);
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

                            }).filter(d -> d != null);
                    dataList.add(data);
                });

        if(dataList.isEmpty()) return null;

        System.out.println("Taking union of datasets...");
        JavaRDD<DataSet> data = dataList.get(0);
        for(int i = 1; i < dataList.size(); i++) {
            data=data.union(dataList.get(i));
            System.out.println("Finished union: "+i);
        }

        return batchBy(data,batchSize,numInputs,numOutputs);
    }

    public static JavaRDD<DataSet> batchBy(JavaRDD<DataSet> data, int batchSize, int numInputs, int numOutputs) {
        data.persist(StorageLevel.MEMORY_AND_DISK());
        long count = data.count();
        System.out.println("Starting count before batching: "+count);

        JavaRDD<DataSet> toReturn = data.repartition((int)count/batchSize).mapPartitions(iter->{
            List<INDArray> labelVecs = new ArrayList<>(batchSize);
            List<INDArray> featureVecs = new ArrayList<>(batchSize);
            for(int i = 0; i < batchSize; i++) {
                if(iter.hasNext()) {
                    DataSet set = iter.next();
                    labelVecs.add(set.getLabels());
                    featureVecs.add(set.getFeatures());
                } else {
                    labelVecs.add(Nd4j.zeros(numOutputs));
                    featureVecs.add(Nd4j.zeros(numInputs));
                }
            }
            return Arrays.asList(new DataSet(Nd4j.vstack(featureVecs),Nd4j.vstack(labelVecs))).iterator();
        }).repartition(200);
        data.unpersist();
        return toReturn;
    }

    public static JavaRDD<DataSet> loadAutoEncoderData(SparkSession spark, int height, int width, int channels, int batch, String... bucketNames) {
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

                }).filter(d->d!=null).repartition(200);

        return batchBy(data,batch,numInputs,numInputs);
    }
}
