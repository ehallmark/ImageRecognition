package main.java.data_loader;

import main.java.flicker_scraper.FlickrScraper;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Evan on 3/21/2017.
 */
public class DataLoader {
    public static Tuple2<List<String>,JavaRDD<DataSet>> loadClassificationData(SparkSession spark, int partitions, int numInputs, String... bucketNames) {
        JavaRDD<Tuple2<String,INDArray>> dataLists = spark.read()
                .format(FlickrScraper.AVRO_FORMAT)
                .load(bucketNames)
                .select("image","category")
                .javaRDD().repartition(partitions)
                .map((Row row) -> {
                    INDArray vec;
                    try {
                        if(row.isNullAt(0) || row.isNullAt(1)) {
                            System.out.println("Row has a null!");
                            return null;
                        }
                        vec = ImageVectorizer.vectorizeImage(ImageIO.read(new ByteArrayInputStream((byte[])(row.get(0)))), numInputs);
                        return new Tuple2<>(row.get(1).toString(),vec);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return null;
                }).filter(d->d!=null);

        List<String> labels = dataLists.map(pair->pair._1).distinct().collect();
        Map<String,Integer> invertedIdxMap = new HashMap<>();
        for(int i = 0; i < labels.size(); i++) {
            invertedIdxMap.put(labels.get(i),i);
        }

        int numOutputs = labels.size();

        JavaRDD<DataSet> data = dataLists.repartition(partitions)
                .map(dataList->{
                    INDArray labelVec = Nd4j.zeros(numOutputs);
                    labelVec.putScalar(invertedIdxMap.get(dataList._1),1.0);
                    return new DataSet(dataList._2,labelVec);
                });

        return new Tuple2<>(labels,data);
    }
}
