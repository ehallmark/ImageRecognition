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
                new File("flickr_urls.txt"),
                new File("C:\\Users\\Evan\\Downloads\\flickr_urls2.txt"),
                new File("C:\\Users\\Evan\\Downloads\\flickr_urls3.txt"),
                new File("C:\\Users\\Evan\\Downloads\\flickr_urls4.txt"),
                new File("C:\\Users\\Evan\\Downloads\\flickr_urls5.txt")

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

        try {
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.MICROSECONDS);
        } catch (Exception e) {
            e.printStackTrace();
        }


        System.out.println("Number of distinct urls: "+urls.size());
        BufferedWriter writer = new BufferedWriter(new FileWriter(mergedFile));
        for(String url : urls) {
            writer.write(url+"\n");
            writer.flush();
        }

        writer.close();
    }
}