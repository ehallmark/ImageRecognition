package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.stream.Collectors;

/**
 * Created by Evan on 3/25/2017.
 */
public class ModelRunner {
    public static void runClassificationModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> trainData, JavaRDD<DataSet> testData, int nEpochs) {
        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(trainData);
            System.out.println("*** Completed epoch {"+i+"} ***");

            System.out.println("Evaluate model....");

            //Perform evaluation (distributed)
            Evaluation evaluation = model.evaluate(testData);
            System.out.println("***** Evaluation *****");
            System.out.println(evaluation.stats());
        }
        System.out.println("****************Model finished********************");
    }

    public static void runClassificationModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data, int nEpochs) {
        //Execute training:
        System.out.println("Splitting data...");
        JavaRDD<DataSet>[] splitSets = data.randomSplit(new double[]{0.25,0.75});
        JavaRDD<DataSet> testData = splitSets[0].repartition(200);
        long testCount = testData.count();
        System.out.println("Test Count: "+testCount);
        JavaRDD<DataSet> trainData = splitSets[1].repartition(200);
        System.out.println("Train count: "+trainData.count());
        runClassificationModel(model,trainData,testData,nEpochs);
    }

    public static void runAutoEncoderModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data, int nEpochs) {
        //Execute training:
        System.out.println("Splitting data...");
        JavaRDD<DataSet>[] splitSets = data.randomSplit(new double[]{0.25,0.75});
        JavaRDD<DataSet> testData = splitSets[0].repartition(200);
        long testCount = testData.count();
        System.out.println("Test Count: "+testCount);
        JavaRDD<DataSet> trainData = splitSets[1].repartition(200);
        System.out.println("Train count: "+trainData.count());
        runAutoEncoderModel(model,trainData,testData,nEpochs);
    }

    public static void runAutoEncoderModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> trainData, JavaRDD<DataSet> testData, int nEpochs) {
        //Execute training:
        //model.setListeners(new ScoreIterationListener(1));
        //Get the variational autoencoder layer
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder autoencoder
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) model.getNetwork().getLayer(0);

        System.out.println("Train model....");
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(trainData);
            System.out.println("*** Completed epoch {"+i+"} ***");
            double overallError = testData.collect().stream().collect(Collectors.averagingDouble(test -> {
                INDArray latentValues = autoencoder.activate(test.getFeatureMatrix(), false);
                INDArray reconstruction = autoencoder.generateAtMeanGivenZ(latentValues);

                double error = 0d;
                for (int r = 0; r < test.getFeatureMatrix().rows(); r++) {
                    double sim = Transforms.cosineSim(test.getFeatureMatrix().getRow(r), reconstruction.getRow(r));
                    if(Double.isNaN(sim)) sim=-1d;
                    error += 1.0 - sim;
                }
                error /= test.getFeatureMatrix().rows();
                return error;
            }));
            System.out.println("Current model score: "+overallError);

        }
        System.out.println("****************Model finished********************");
    }
}
