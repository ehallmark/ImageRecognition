package main.java.flicker_scraper;

import org.apache.commons.io.FileUtils;

import java.io.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/18/2017.
 */
public class ReadAndSaveFileListFromGCS {
    public static File IMAGE_LOCATIONS_FILE = new File("/mnt/bucket/actual_image_locations.txt");
    public static void main(String[] args) throws Exception {
        File dir = new File(ScrapeImages.IMAGE_DIR);
        File urls = MergeUrlFiles.mergedFile;
        BufferedReader reader = new BufferedReader(new FileReader(urls));
        AtomicInteger cnt = new AtomicInteger(0);
        BufferedWriter writer = new BufferedWriter(new FileWriter(IMAGE_LOCATIONS_FILE));
        reader.lines().limit(4000000).forEach(line->{
            int hash = line.hashCode();
            String url = ScrapeImages.IMAGE_DIR+String.valueOf(hash)+".jpg";
            if(new File(url).exists()) {
                try {
                    writer.write(url + "\n");
                    writer.flush();
                    System.out.println(cnt.getAndIncrement());
                } catch(Exception e) {
                    e.printStackTrace();
                }
            }
        });

        System.out.println("Finished");
        writer.close();
        reader.close();
    }
}
