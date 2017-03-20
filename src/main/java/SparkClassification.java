package main.java;

import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
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
import scala.Tuple2;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
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
        sparkConf.setAppName("Image Recognition Autoencoder");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        // Algorithm
        List<String> bucketNames = sc.textFile("gs://image-scrape-dump/all_countries.txt").distinct().collect();

        int batch = 1;
        int rows = 40;
        int cols = 40;
        int channels = 3;
        int numInputs = rows*cols*channels;
        int nEpochs = 2000;
        int partitions = 50;

        List<Tuple2<Integer,INDArray>> dataLists = new ArrayList<>();
        AtomicInteger bucketIdx = new AtomicInteger(0);
        bucketNames.forEach(bucket->{
            try {
                System.out.println("Trying bucket: "+bucket);
                int idx = bucketIdx.get();
                List<Tuple2<Integer,INDArray>> dataSets = sc.objectFile("gs://image-scrape-dump/labeled_images/" + bucket.toLowerCase().replaceAll("[^a-z0-9- ]","").replaceAll(" ","_").trim(), 50)
                        .map((tup) -> {
                            INDArray vec;
                            try {
                                vec = ImageVectorizer.vectorizeImage(ImageIO.read(new ByteArrayInputStream(((Tuple2<String,byte[]>)tup)._2)), numInputs);
                                System.out.println("Vec: "+vec);
                                return new Tuple2<>(idx,vec);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                            return null;
                        }).filter(d->d!=null).collect();
                if(dataSets.size()>0) {
                    bucketIdx.getAndIncrement();
                    dataLists.addAll(dataSets);
                }

            } catch(Exception e) {
            }
        });
        int numOutputs = bucketIdx.get();
        System.out.println("DataList size: "+dataLists.size());
        JavaRDD<DataSet> data = sc.parallelize(dataLists)
                .map(dataList->{
                    INDArray labelVec = Nd4j.zeros(numOutputs);
                    labelVec.putScalar(dataList._1,1.0);
                    return new DataSet(dataList._2,labelVec);
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
