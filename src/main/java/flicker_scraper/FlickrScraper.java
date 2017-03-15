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
    private static final int timeout = 5000;
    private static final int maxRetriesPerPage = 1;
    private static final AtomicInteger totalUrlCounter = new AtomicInteger(0);
    public static void writeImageUrlsFromSearchText(String searchText, Set<Integer> alreadyContains) {
        Document doc;
            boolean shouldContinue = true;
            int page = 1;
            int numRetriesOnCurrentPage = 0;
            while(shouldContinue) {
                shouldContinue=false;
                String searchURL = "https://www.flickr.com/search/?text=" + searchText + "&page=" + page;
                try {
                    doc = Jsoup.connect(searchURL).timeout(timeout).get();

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
                                    if(!alreadyContains.contains(url.hashCode())) {
                                        if(ScrapeImages.trySaveImageToGoogleCloud(url)) {
                                            shouldContinue = true;
                                            alreadyContains.add(url.hashCode());
                                            totalUrlCounter.getAndIncrement();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    page++;
                    numRetriesOnCurrentPage=0;
                    System.out.println("Search: "+searchText);
                    System.out.println("Page: "+page);
                    System.out.println("Images ingested so far: "+totalUrlCounter.get());
                } catch(Exception e) {
                    System.out.println("Error: "+e.getMessage());
                    System.out.println(searchURL);
                    numRetriesOnCurrentPage++;
                    if(e instanceof SocketTimeoutException) {
                        if(numRetriesOnCurrentPage<maxRetriesPerPage) {
                            shouldContinue=true;
                        }
                    }
                }
            }
    }

    public static void main(String[] args) throws Exception{
        // test
        BufferedReader reader = new BufferedReader(new FileReader(new File("search_words.txt")));
        try {
            System.out.println("Starting to clean URLs");
            Set<Integer> alreadyContains = new HashSet<>();
            System.out.println("Finished cleaning URLs");
            ForkJoinPool pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors()*4);
            AtomicInteger cnt = new AtomicInteger(0);
            reader.lines().forEach(line -> {
                pool.execute(new RecursiveAction() {
                    @Override
                    protected void compute() {
                        writeImageUrlsFromSearchText(line.split(",")[0].trim(), alreadyContains);
                        System.out.println(cnt.getAndIncrement());
                    }
                });
            });
            pool.shutdown();
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        } finally {
            reader.close();
        }
    }
}
