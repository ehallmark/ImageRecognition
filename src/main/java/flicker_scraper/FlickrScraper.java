package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;
import main.java.image_vectorization.ImageVectorizer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.berkeley.Pair;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import scala.Tuple2;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.util.*;
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
    public static List<String> writeImageUrlsFromSearchText(String searchText) {
        Document doc;
        boolean shouldContinue = true;
        int page = 1;
        int numRetriesOnCurrentPage = 0;
        List<String> list = new ArrayList<>();
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
                                if (!url.endsWith("_s.jpg") && url.length() > 10) url = url.substring(0, url.length() - 6) + "_s.jpg";
                                shouldContinue = true;
                                totalUrlCounter.getAndIncrement();
                                list.add(url);
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
        return list;
    }

    public static void main(String[] args) throws Exception{
        // test
        boolean useSparkLocal = false;
        int numPartitions = 1;
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("FlickrScraper");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        List<Tuple2<String,List<String>>> urls = sc.textFile("gs://image-scrape-dump/all_countries.txt").map(line-> {
            String term = line.split("[,\\[\\]()]")[0].replaceAll("[^a-zA-z0-9- ]", "").trim().toLowerCase();
            return term;
        }).distinct().map(term->{
            return new Tuple2<>(term,writeImageUrlsFromSearchText(term));
        }).filter(tup->tup._2.size()>0).collect();
        System.out.println("Finished collecting urls... Now loading images");
        urls.forEach(pair->{
            JavaPairRDD<String,byte[]> data = sc.parallelize(pair._2,50).mapToPair(url->{
                ByteArrayOutputStream baos = null;
                try {
                    baos = new ByteArrayOutputStream();
                    ImageIO.write(ImageStreamer.loadImage(new URL(url)), "jpg", baos);
                    return new Tuple2<>(pair._1,baos.toByteArray());
                } catch(Exception e) {
                }
                finally {
                    try {
                        baos.close();
                    } catch (Exception e) {
                    }
                }
                return null;

            }).filter(image->image!=null).repartition(numPartitions);
            long count = data.count();
            if(count>0) {
                System.out.println("Num urls for: "+pair._1+", "+count);
                try {
                    data.saveAsObjectFile("gs://image-scrape-dump/labeled-images/" + pair._1.replaceAll(" ", "_"));
                } catch(Exception e) {
                    e.printStackTrace();
                }
            }
        });
        System.out.println("Finished saving");
        System.out.println("Num urls: "+urls.size());
    }
}
