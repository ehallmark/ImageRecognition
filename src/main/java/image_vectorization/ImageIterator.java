package main.java.image_vectorization;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.net.URLEncoder;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/11/2017.
 */
public class ImageIterator implements DataSetIterator{
    public static final File urlFile = new File("fall11_urls.txt");
    private volatile Iterator<Pair<Integer,INDArray>> urlIterator;
    private List<Pair<Integer,INDArray>> urlList;
    private int totalInputs;
    private int batch;

    public ImageIterator(int batch,int numInputs, int limit) {
        this.batch=batch;
        this.totalInputs=numInputs;
        urlList=new ArrayList<>();
        AtomicInteger idCounter = new AtomicInteger(0);
        LineSentenceIterator iter = new LineSentenceIterator(urlFile);
        while(iter.hasNext()&&idCounter.get()<limit) {
            String line = iter.nextSentence();
            if(line==null) continue;
            if(!line.contains("flickr")||!line.endsWith(".jpg"))continue;
            int idx = line.indexOf("http:");
            if(idx>=0) {
                String urlString = line.substring(idx);
                urlString=urlString.substring(0,urlString.length()-4)+"_s.jpg"; // gets small images
                System.out.println(urlString);
                try {
                    URL url = new URL(urlString);
                    BufferedImage image = ImageStreamer.loadImage(url);
                    if(image==null)continue;
                    INDArray f = ImageVectorizer.vectorizeImage(image,totalInputs,true);
                    urlList.add(new Pair<>(idCounter.getAndIncrement(), f));
                    System.out.println(idCounter.get());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        }
        reset();
    }

    @Override
    public DataSet next(int num) { // only does one at a time no matter what for now
        int cnt = 0;
        INDArray features = Nd4j.create(num,totalInputs);
        while(urlIterator.hasNext()&&cnt<num) {
            Pair<Integer,INDArray> pair = urlIterator.next();
            features.putRow(cnt,pair.getSecond());
            cnt++;
        }
        return new DataSet(features,features);
    }

    @Override
    public int totalExamples() {
        return urlList.size();
    }

    @Override
    public int inputColumns() {
        return totalInputs;
    }

    @Override
    public int totalOutcomes() {
        return totalInputs;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        urlIterator=urlList.iterator();
    }

    @Override
    public int batch() {
        return batch;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException("cursor");
    }

    @Override
    public int numExamples() {
        return urlList.size();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return urlIterator.hasNext();
    }

    @Override
    public DataSet next() {
        return next(batch());
    }


    public static void main(String[] args) throws Exception {
        int batch = 10;
        int limit = 1000;
        int rows = 40;
        int cols = 40;
        int channels = 1;
        int numInputs = rows*cols*channels;
        int nEpochs = 2000;
        // 1,4 billion!
        ImageIterator iterator = new ImageIterator(batch,numInputs,limit);

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

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            while(iterator.hasNext())model.fit(iterator.next());
            iterator.reset();
            System.out.println("*** Completed epoch {"+i+"} ***");

            System.out.println("Evaluate model....");
            double error = 0;
            while(iterator.hasNext()){
                DataSet ds = iterator.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                for(int r = 0; r < output.rows(); r++) {
                    double sim = Transforms.cosineSim(output.getRow(r),ds.getFeatureMatrix().getRow(r));
                    if(new Double(sim).isNaN()) error+=2.0;
                    else error+= 1.0-sim;
                }
            }
            System.out.println("Error: "+error);
            iterator.reset();
        }
        System.out.println("****************Example finished********************");


    }
}
