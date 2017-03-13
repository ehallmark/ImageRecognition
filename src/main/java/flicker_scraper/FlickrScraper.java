package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;
import main.java.image_vectorization.ImageVectorizer;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;

import java.io.*;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/12/2017.
 */
public class FlickrScraper {
    private static final int timeout = 10000;
    public static Collection<String> getImageUrlsFromSearchText(String searchText) {
        Set<String> urls = new HashSet<>();
        Document doc;
            boolean shouldContinue = true;
            int page = 1;
            while(shouldContinue) {
                shouldContinue=false;

                try {
                    doc = Jsoup.connect("https://www.flickr.com/search/?text=" + searchText + "&page=" + page).timeout(timeout).get();

                    // get all links
                    //Elements divs = doc.select("div[background-image]");
                    Elements divs = doc.select("div.main div.photo-list-photo-view");
                    for (Element div : divs) {
                        // get the value from href attribute
                        String style = div.attr("style");
                        if (style != null && style.contains("background-image:")) {
                            String url = style.substring(style.indexOf("background-image:"));
                            int start = url.indexOf("(") + 1;
                            int end = url.indexOf(")");
                            if (end > start && start > 0) {
                                url = url.substring(start, end);
                                if (!url.isEmpty()) {
                                    if (!url.startsWith("http")) url = "http:" + url;
                                    if (!url.endsWith("_s.jpg") && url.length() > 10)
                                        url = url.substring(0, url.length() - 6) + "_s.jpg";
                                    shouldContinue = true;
                                    urls.add(url);
                                }
                            }
                        }
                    }
                    page++;
                } catch(Exception e) {
                    System.out.println("Error");
                    if(e instanceof SocketTimeoutException) {
                        shouldContinue=true;
                    }
                }
            }
        return urls;
    }

    public static void main(String[] args) throws Exception{
        // test
        File flickrFile = new File("flickr_urls.txt");
        Set<String> existingUrls = new HashSet<>();
        System.out.println("Checking for existing URLs");
        if(flickrFile.exists()) {
            // read in values
            BufferedReader reader = new BufferedReader(new FileReader(flickrFile));
            reader.lines().forEach(line->{
                existingUrls.add(line);
            });
        }
        System.out.println("Total URLs so far: "+existingUrls.size());
        BufferedWriter writer = new BufferedWriter(new FileWriter(flickrFile));
        BufferedReader reader = new BufferedReader(new FileReader(new File("search_words.txt")));
        System.out.println("Starting to clean URLs");
        Set<Integer> alreadyContains = new HashSet<>();
        existingUrls.forEach(url->{
            alreadyContains.add(url.hashCode());
            try {
                writer.write(url + "\n");
            } catch(Exception e) {
                e.printStackTrace();
            }
        });
        System.out.println("Finished cleaning URLs");
        ForkJoinPool pool = new ForkJoinPool();
        AtomicInteger cnt = new AtomicInteger(0);
        reader.lines().forEach(line->{
            pool.execute(new RecursiveAction() {
                @Override
                protected void compute() {
                    getImageUrlsFromSearchText(line).forEach(url-> {
                        try {
                            Integer hash = url.hashCode();
                            if (!alreadyContains.contains(hash)) {
                                //if (ImageStreamer.loadImage(new URL(url)) != null) {
                                writer.write(url+"\n");
                                writer.flush();
                                //} too slow i think
                                alreadyContains.add(hash);
                            }
                        } catch(Exception e) {
                            e.printStackTrace();
                        }

                    });
                    System.out.println(cnt.getAndIncrement());
                }
            });
        });
        pool.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        writer.flush();
        writer.close();

    }
}
