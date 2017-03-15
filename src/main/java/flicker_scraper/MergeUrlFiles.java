package main.java.flicker_scraper;

import java.io.*;
import java.time.LocalDate;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/14/2017.
 */
public class MergeUrlFiles {
    public static File mergedFile = new File("all_flickr_urls.txt");

    public static void main(String[] args) throws IOException {
        File[] files = new File[]{
                new File("flickr_urls.txt"),
                new File("flickr_urls2.txt"),
                new File("flickr_urls3.txt"),
                new File("flickr_urls4.txt"),
                new File("flickr_urls5.txt"),
        };
        AtomicInteger count = new AtomicInteger(0);
        Set<String> urls = (new HashSet<>());
        for(File file : files) {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            reader.lines().forEach(line->{
                urls.add(line);
                System.out.println(count.getAndIncrement());
            });
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