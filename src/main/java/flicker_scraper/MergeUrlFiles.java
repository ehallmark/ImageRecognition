package main.java.flicker_scraper;

import java.io.*;
import java.time.LocalDate;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;

/**
 * Created by Evan on 3/14/2017.
 */
public class MergeUrlFiles {
    public static void main(String[] args) throws IOException {
        File mergedFile = new File("all_flickr_urls_"+ LocalDate.now().toString()+".txt");
        File[] files = new File[]{

        };

        ForkJoinPool pool = new ForkJoinPool(files.length);
        Set<String> urls = Collections.synchronizedSet(new HashSet<>());
        for(File file : files) {
            pool.execute(new RecursiveAction() {
                @Override
                protected void compute() {
                    try {
                        BufferedReader reader = new BufferedReader(new FileReader(file));
                        reader.lines().forEach(line->{
                            urls.add(line);
                        });
                    }catch(Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }

        BufferedWriter writer = new BufferedWriter(new FileWriter(mergedFile));
        for(String url : urls) {
            writer.write(url+"\n");
            writer.flush();
        }

        writer.close();
        try {
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.MICROSECONDS);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}