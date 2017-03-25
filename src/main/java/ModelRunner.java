package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by Evan on 3/25/2017.
 */
public class ModelRunner {
    public static void runClassificationModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data, int nEpochs) {
        //Execute training:
        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(data);
            System.out.println("*** Completed epoch {"+i+"} ***");

            System.out.println("Evaluate model....");

            //Perform evaluation (distributed)
            Evaluation evaluation = model.evaluate(data);
            System.out.println("***** Evaluation *****");
            System.out.println(evaluation.stats());
        }
        System.out.println("****************Model finished********************");
    }

    public static void runAutoEncoderModel(SparkDl4jMultiLayer model, JavaRDD<DataSet> data, int nEpochs) {
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
        System.out.println("****************Model finished********************");
    }
}
