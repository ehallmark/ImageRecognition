package main.java.flicker_scraper;

import main.java.image_vectorization.ImageStreamer;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

/**
 * Created by Evan on 3/15/2017.
 */
public class ScrapeImages {
    public static final String IMAGE_DIR = "/mnt/bucket/images/";
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(MergeUrlFiles.mergedFile));
        Stream<String> lines = reader.lines();
        if(args.length>0) {
            int offset = Integer.valueOf(args[0]);
            System.out.println("Offset specified: "+offset);
            lines = lines.skip(offset);
        }
        if(args.length>1) {
            int limit = Integer.valueOf(args[1]);
            System.out.println("Limit specified: "+limit);
            lines = lines.limit(limit);
        }
        AtomicInteger cnt = new AtomicInteger(0);
        int numProcessors = Runtime.getRuntime().availableProcessors()*2;
        int numThreads = 8;
        ForkJoinPool pool = new ForkJoinPool(numProcessors);
        List<Thread> threads = new ArrayList<>(numThreads);
        lines.forEach(line->{
            Thread thread = new Thread() {
                public void run() {
                    if(trySaveImageToGoogleCloud(line)) {
                        System.out.println(cnt.getAndIncrement());
                    }
                }
            };
            threads.add(thread);
            if(threads.size()>=numThreads) {
                List<Thread> taskThreads = new ArrayList<>(threads);
                threads.clear();
                pool.execute(new RecursiveAction() {
                    @Override
                    protected void compute() {
                        taskThreads.forEach(t->t.start());
                        taskThreads.forEach(t->{
                            try {
                                t.join();
                            } catch(Exception e) {
                                e.printStackTrace();
                            }
                        });
                    }
                });
            }

            if(pool.getQueuedSubmissionCount()>2*numProcessors) {
                pool.awaitQuiescence(1000,TimeUnit.MILLISECONDS);
            }
        });
        pool.shutdown();
        pool.awaitTermination(Long.MAX_VALUE, TimeUnit.MICROSECONDS);
    }

    public static boolean trySaveImageToGoogleCloud(String urlString) {
        try {
            BufferedImage image = ImageStreamer.loadImage(new URL(urlString));
            if(image!=null) {
                File file = new File(IMAGE_DIR+urlString.hashCode()+".jpg");
                if(!file.exists()) {
                    ImageIO.write(image, "jpg", file);
                    return true;
                }
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
        return false;
    }
}
