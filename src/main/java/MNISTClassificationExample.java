package main.java;

import main.java.data_loader.DataLoader;
import main.java.flicker_scraper.IngestMNIST;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

/**
 * Created by Evan on 4/2/2017.
 */
public class MNISTClassificationExample {
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
        int numOutputs = 10; // Number of digits
        int nEpochs = 100;

        JavaRDD<DataSet> data = IngestMNIST.getTrainData(spark,batch,numInputs,numOutputs);
        JavaRDD<DataSet> test = IngestMNIST.getTestData(spark,batch,numInputs,numOutputs);

        System.out.println("DataList size: "+data.count());

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(69)
                .iterations(1) // Training iterations as above
                .miniBatch(true)
                .regularization(true).l2(0.0005)
                .learningRate(.005)
                .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(rows,cols,channels)) //See note below
                .backprop(true).pretrain(false).build();

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batch)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(5)
                .workerPrefetchNumBatches(1)            //Async prefetching: 2 examples per worker
                .batchSizePerWorker(batch)
                .build();

        //Create the Spark network
        SparkDl4jMultiLayer model = new SparkDl4jMultiLayer(sc, conf, tm);

        ModelRunner.runClassificationModel(model, data, test, nEpochs);

    }
}
