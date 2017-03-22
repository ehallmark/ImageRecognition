package main.java.flicker_scraper;

import java.io.Serializable;

/**
 * Created by ehallmark on 3/20/17.
 */
public class Image implements Serializable {
    private static final long serialVersionUID = 69L;
    private String category;
    private byte[] image;

    public String getCategory() {
        return category;
    }

    public byte[] getImage() {
        return image;
    }

    public void setCategory(String category) {
        this.category=category;
    }

    public void setImage(byte[] image) {
        this.image=image;
    }
}
