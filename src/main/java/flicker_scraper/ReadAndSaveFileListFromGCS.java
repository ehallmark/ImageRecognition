package main.java.flicker_scraper;

import org.apache.commons.io.FileUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/18/2017.
 */
public class ReadAndSaveFileListFromGCS {
    public static File IMAGE_LOCATIONS_FILE = new File("/mnt/bucket/actual_image_locations.txt");
    public static void main(String[] args) throws Exception {
        File dir = new File(ScrapeImages.IMAGE_DIR);
        AtomicInteger cnt = new AtomicInteger(0);
        BufferedWriter writer = new BufferedWriter(new FileWriter(IMAGE_LOCATIONS_FILE));
        FileUtils.iterateFiles(dir,new String[]{"jpg"},false).forEachRemaining(file->{
            try {
                writer.write(file.getName() + "\n");
                writer.flush();
                System.out.println(cnt.getAndIncrement());
            } catch(Exception e) {
                e.printStackTrace();
            }
        });
        System.out.println("Finished");
        writer.close();
    }
}
