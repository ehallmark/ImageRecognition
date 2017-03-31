package main.java;

import edu.stanford.nlp.io.StringOutputStream;
import main.java.data_loader.DataLoader;
import main.java.flicker_scraper.ReadAndSaveFileListFromGCS;
import main.java.image_vectorization.ImageIterator;
import main.java.image_vectorization.ImageStreamer;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
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
import java.io.File;
import java.net.URL;
import java.util.*;
import java.util.stream.StreamSupport;

/**
 * Created by Evan on 3/18/2017.
 */
public class SparkAutoEncoder {
    public static void main(String[] args) throws Exception {
        // Spark stuff
        boolean useSparkLocal = false;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("Image Recognition AutoEncoder");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        SparkSession spark = SparkSession.builder()
                .appName("FlickrScraper")
                .getOrCreate();

        if(args.length<1) {
            throw new RuntimeException("Must specify databucket argument");
        }

        // Algorithm

        int batch = 10;
        int rows = 28;
        int cols = 28;
        int channels = 3;
        int numInputs = rows*cols*channels;
        int nEpochs = 20;

        int vectorSize = 20;

        String dataBucketName = "gs://image-scrape-dump/labeled-images/"+args[0];
        JavaRDD<DataSet> data = DataLoader.loadAutoEncoderData(spark,rows,cols,channels,batch,dataBucketName);

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(69)
                .iterations(3) // Training iterations as above
                .miniBatch(true)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .learningRate(.01)
                .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new AutoEncoder.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.RELU)
                        .nIn(numInputs)
                        .nOut((numInputs*3)/4)
                        .build())
                .layer(1, new AutoEncoder.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.RELU)
                        .nIn((numInputs*3)/4)
                        .nOut(vectorSize*2)
                        .build())
                .layer(2, new AutoEncoder.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.RELU)
                        .nIn(vectorSize*2)
                        .nOut(vectorSize)
                        .build())
                .layer(3, new AutoEncoder.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.RELU)
                        .nIn(vectorSize)
                        .nOut(vectorSize*2)
                        .build())
                .layer(4, new AutoEncoder.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.RELU)
                        .nIn(vectorSize*2)
                        .nOut((numInputs*3)/4)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn((numInputs*3)/4)
                        .nOut(numInputs)
                        .activation(Activation.SIGMOID)
                        .build())
                .backprop(true).pretrain(true).build();

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batch)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(5)
                .workerPrefetchNumBatches(1)            //Async prefetching: 2 examples per worker
                .batchSizePerWorker(batch)
                .build();

        //Create the Spark network
        SparkDl4jMultiLayer model = new SparkDl4jMultiLayer(sc, conf, tm);
        ModelRunner.runAutoEncoderModel(model,data,nEpochs);

    }
}
