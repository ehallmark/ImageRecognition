package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
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
import java.time.LocalDate;
import java.util.*;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Evan on 3/12/2017.
 */
public class FlickrScraper {
    private static final int timeout = 5000;
    public static final String AVRO_FORMAT = "com.databricks.spark.avro";
    public static final String LABELED_IMAGES_BUCKET = "gs://image-scrape-dump/labeled-images/";
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
        if(args.length<2) throw new RuntimeException("Must specify args [1] searchWordFile and [2] bucketToSaveIn");
        String searchWordFile = args[0]; // "gs://image-scrape-dump/top_us_cities.txt";
        String bucketToSaveIn = args[1]; // "us-cities";
        boolean useSparkLocal = false;
        int numPartitions = 60;

        SparkConf sparkConf = new SparkConf();


        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("FlickrScraper");
        SparkSession spark = SparkSession.builder()
                .appName("FlickrScraper")
                .config(sparkConf)
                .getOrCreate();

        JavaSparkContext sc = JavaSparkContext.fromSparkContext(spark.sparkContext());

        JavaRDD<Image> data = sc.textFile(searchWordFile).map(line-> {
            String term = line.split("[,\\[\\]()]")[0].replaceAll("[^a-zA-z0-9- ]", "").trim().toLowerCase();
            return term;
        }).distinct().repartition(numPartitions).map(term->{
            return new Tuple2<>(term,writeImageUrlsFromSearchText(term));
        }).filter(tup->tup._2.size()>0).repartition(numPartitions)
                .flatMap(tup->{
                    return tup._2.stream().map(url->{
                        ByteArrayOutputStream baos = null;
                        try {
                            baos = new ByteArrayOutputStream();
                            BufferedImage img = ImageStreamer.loadImage(new URL(url));
                            if(img!=null) {
                                ImageIO.write(img, "jpg", baos);
                                Image myImage = new Image();
                                myImage.setCategory(tup._1);
                                myImage.setImage(baos.toByteArray());
                                return myImage;
                            }
                        } catch(Exception e) {
                        }
                        finally {
                            try {
                                baos.close();
                            } catch (Exception e) {
                            }
                        }
                        return null;

                    }).filter(image->image!=null).iterator();
                });
        System.out.println("Finished collecting images... Now saving images");

        long count = data.count();
        if(count>0) {
            System.out.println("Num urls: "+count);
            try {
                Dataset<Row> dataset = spark.createDataFrame(data,Image.class);
                dataset.write()
                        .format(AVRO_FORMAT)
                        .save(LABELED_IMAGES_BUCKET+bucketToSaveIn);
            } catch(Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println("Finished saving");
    }
}
