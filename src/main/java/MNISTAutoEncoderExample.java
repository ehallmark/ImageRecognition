package main.java;

import main.java.flicker_scraper.IngestMNIST;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by Evan on 4/2/2017.
 */
public class MNISTAutoEncoderExample {
    public static void main(String[] args) {
        boolean useSparkLocal = false;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("Image Recognition Classification");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        SparkSession spark = SparkSession.builder()
                .appName("FlickrScraper")
                .getOrCreate();

        int batch = 100;
        int rows = 28;
        int cols = 28;
        int channels = 1;
        int numInputs = rows*cols*channels;
        int nEpochs = 100;
        int vectorSize = 50;

        JavaRDD<DataSet> data = IngestMNIST.getTrainAutoEncoderData(spark,batch,numInputs);
        JavaRDD<DataSet> test = IngestMNIST.getTestAutoEncoderData(spark,batch,numInputs);

        System.out.println("DataList size: "+data.count());

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(69)
                .iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .updater(Updater.RMSPROP).rmsDecay(0.95)
                .weightInit(WeightInit.XAVIER)
                .regularization(true).l2(1e-4)
                .gradientNormalizationThreshold(1.0)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .miniBatch(true)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .pzxActivationFunction(Activation.IDENTITY)
                        //.dropOut(0.5)
                        .encoderLayerSizes(200,200,200)
                        .decoderLayerSizes(200,200,200)
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
                        .nIn(numInputs)                       //Input size: 28x28
                        .nOut(vectorSize)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
                        .build())
                .pretrain(true).backprop(false).build();

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batch)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(5)
                .workerPrefetchNumBatches(1)            //Async prefetching: 2 examples per worker
                .batchSizePerWorker(batch)
                .build();

        //Create the Spark network
        SparkDl4jMultiLayer model = new SparkDl4jMultiLayer(sc, conf, tm);

        ModelRunner.runAutoEncoderModel(model,data,test,nEpochs);

    }
}
