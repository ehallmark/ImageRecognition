package main.java;

import main.java.flicker_scraper.ReadAndSaveFileListFromGCS;
import main.java.image_vectorization.ImageIterator;
import main.java.image_vectorization.ImageStreamer;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
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

import java.io.File;
import java.net.URL;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.StreamSupport;

/**
 * Created by Evan on 3/18/2017.
 */
public class SparkAutoEncoder {
    public static void main(String[] args) throws Exception {
        // Spark stuff
        File imageLocationsFile = new File(ReadAndSaveFileListFromGCS.IMAGE_LOCATIONS_FILE);
        boolean useSparkLocal = true;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("Image Recognition Autoencoder");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        // Algorithm

        int batch = 10;
        int rows = 40;
        int cols = 40;
        int channels = 3;
        int numInputs = rows*cols*channels;
        int nEpochs = 2000;
        int partitions = 1000;

        JavaRDD<DataSet> data = sc.textFile(imageLocationsFile.getAbsolutePath(),partitions)
                .mapPartitions((Iterator<String> iter) -> {
                    Random rand = new Random();
                    INDArray features = Nd4j.create(batch,numInputs);
                    for(int i = 0; i < batch; i++) {
                        INDArray vec = null;
                        if(iter.hasNext()) {
                            try {
                                vec = ImageVectorizer.vectorizeImage(ImageStreamer.loadImage(new URL("http://storage.googleapis.com/image-scrape-dump/images/"+iter.next())), numInputs);
                            } catch(Exception e) {

                            }
                        }
                        if(vec==null) {
                            for(int j = 0; j < numInputs; j++) {
                                features.putScalar(i,j,rand.nextDouble());
                            }
                        } else {
                            features.putRow(i,vec);
                        }
                    }
                    return Arrays.asList(new DataSet(features,features)).iterator();
        });

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
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(numInputs)
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

        //Execute training:
        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(data);
            System.out.println("*** Completed epoch {"+i+"} ***");

            System.out.println("Evaluate model....");
            double totalError = data.map(ds->{
                INDArray output = model.getNetwork().output(ds.getFeatureMatrix(), false);
                double error = 0;
                for(int r = 0; r < output.rows(); r++) {
                    double sim = Transforms.cosineSim(output.getRow(r),ds.getFeatureMatrix().getRow(r));
                    if(new Double(sim).isNaN()) error+= 2.0;
                    else error+= 1.0-sim;
                }
                return error;
            }).reduce((a,b)->a+b);
            System.out.println("Error: "+totalError);
        }
        System.out.println("****************Example finished********************");

    }
}
