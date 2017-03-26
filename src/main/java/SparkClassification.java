package main.java;

import main.java.data_loader.DataLoader;
import main.java.flicker_scraper.FlickrScraper;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.TypedColumn;
import org.apache.spark.sql.execution.columnar.ByteArrayColumnType;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import scala.Tuple2;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/18/2017.
 */
public class SparkClassification {
    public static void main(String[] args) throws Exception {
        // Spark stuff
        boolean useSparkLocal = true;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("Image Recognition Classification");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        SparkSession spark = SparkSession.builder()
                .appName("FlickrScraper")
                .getOrCreate();

        if(args.length<2) {
            throw new RuntimeException("Must specify filename and databucket name");
        }
        // Algorithm
        String fileName = "gs://image-scrape-dump/"+args[0];
        String dataBucketName = "gs://image-scrape-dump/labeled-images/"+args[1];

        int batch = 1;
        int rows = 32;
        int cols = 32;
        int channels = 3;
        int numInputs = rows*cols*channels;
        int nEpochs = 2000;

        List<String> labels = sc.textFile(fileName).map(line->{
            return line.split("[,\\[\\]()]")[0].replaceAll("\\s+"," ").replaceAll("[^a-zA-z0-9- ]", "").trim().toLowerCase();
        }).distinct().collect();

        JavaRDD<DataSet> data = DataLoader.loadClassificationData(spark,rows,cols,channels,labels,false,dataBucketName);
        int numOutputs = labels.size();

        System.out.println("DataList size: "+data.count());
        System.out.println("Labels size: "+labels.size());

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(69)
                .iterations(3) // Training iterations as above
                .miniBatch(true)
                .regularization(true).l2(0.0005)
                .learningRate(.01).biasLearningRate(0.02)
                .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut((numInputs+numOutputs)/2)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut((numInputs+numOutputs+numOutputs+numOutputs)/4)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(numOutputs).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(numOutputs)
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutionalFlat(rows,cols,channels)) //See note below
                .backprop(true).pretrain(false).build();

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batch)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(5)
                .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
                .batchSizePerWorker(batch)
                .build();

        //Create the Spark network
        SparkDl4jMultiLayer model = new SparkDl4jMultiLayer(sc, conf, tm);

        ModelRunner.runClassificationModel(model,data,nEpochs);

    }
}
