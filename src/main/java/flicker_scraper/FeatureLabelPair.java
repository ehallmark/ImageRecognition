package main.java.flicker_scraper;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by Evan on 4/2/2017.
 */
public class FeatureLabelPair implements Serializable {
    private static final long serialVersionUID = 69L;

    private float[] features;
    private float[] labels;

    public FeatureLabelPair() {

    }

    public float[] getFeatures() {return features;}
    public float[] getLabels() { return labels; }
    public void setFeatures(float[] features) { this.features=features;}
    public void setLabels(float[] labels) {this.labels=labels;}
}
