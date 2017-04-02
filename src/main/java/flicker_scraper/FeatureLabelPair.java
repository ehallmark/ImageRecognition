package main.java.flicker_scraper;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Evan on 4/2/2017.
 */
public class FeatureLabelPair {
    private INDArray features;
    private INDArray labels;
    public FeatureLabelPair(INDArray features, INDArray labels) {
        this.features=features;
        this.labels=labels;
    }

    public INDArray getFeatures() {return features;}
    public INDArray getLabels() { return labels; }
    public void setFeatures(INDArray features) { this.features=features;}
    public void setLabels(INDArray labels) {this.labels=labels;}
}