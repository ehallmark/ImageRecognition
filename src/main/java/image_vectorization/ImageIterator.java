package main.java.image_vectorization;

import com.sun.javafx.binding.StringFormatter;
import org.apache.commons.codec.StringEncoder;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
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
    private volatile Iterator<Pair<Integer,URL>> urlIterator;
    private List<Pair<Integer,URL>> urlList;
    private int totalInputs;
    private int batch;

    public ImageIterator(int batch,int numInputs, int limit, boolean train) {
        this.batch=batch;
        this.totalInputs=numInputs;
        urlList=new ArrayList<>();
        AtomicInteger idCounter = new AtomicInteger(0);
        LineSentenceIterator iter = new LineSentenceIterator(urlFile);
        Random rand = new Random(69);
        while(iter.hasNext()&&idCounter.get()<limit) {
            if(rand.nextBoolean()) {
                if(train)continue;
            } else {
                if(!train)continue;
            }
            String line = iter.nextSentence();
            if(line==null) continue;
            int idx = line.indexOf("http:");
            if(idx>=0) {
                String urlString = line.substring(idx);
                System.out.println(urlString);
                try {
                    URL url = new URL(urlString);
                    urlList.add(new Pair<>(idCounter.getAndIncrement(), url));
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
        BufferedImage image;
        int cnt = 0;
        INDArray features = Nd4j.create(num,totalInputs);
        INDArray labels = Nd4j.zeros(num,totalOutcomes());
        while(urlIterator.hasNext()&&cnt<num) {
            Pair<Integer,URL> pair = urlIterator.next();
            image = ImageStreamer.loadImage(pair.getSecond());
            if(image==null)continue;
            INDArray f = ImageVectorizer.vectorizeImage(image,totalInputs);
            labels.putScalar(cnt,pair.getFirst(),1.0);
            features.putRow(cnt,f);
            cnt++;
        }
        return new DataSet(features,labels);
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
        return urlList.size();
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
        int batch = 5;
        int limit = 100;
        int numInputs = 100;
        int nEpochs = 20;
        // 1,4 billion!
        ImageIterator iterator = new ImageIterator(batch,numInputs,limit,true);
        ImageIterator testIterator = new ImageIterator(batch,numInputs,limit,false);
                /*
            Construct the neural network
         */
        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(69)
                .iterations(3) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(numInputs)
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
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(limit)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)
        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        System.out.println("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(iterator);
            System.out.println("*** Completed epoch {"+i+"} ***");

            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(limit);
            while(testIterator.hasNext()){
                DataSet ds = testIterator.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);

            }
            System.out.println(eval.stats());
            testIterator.reset();
        }
        System.out.println("****************Example finished********************");


    }
}
