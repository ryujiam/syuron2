import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.job.RecommenderJob;
import net.librec.recommender.cf.ranking.BPRRecommender;
import parameter.gridSearch;
import recommend.EfmRecommender;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        testRecommender();
        //gridtest();

    }

    public static void testRecommender() throws ClassNotFoundException, LibrecException, IOException {
        Configuration conf = new Configuration();
        Configuration.Resource resource = new Configuration.Resource("efm-test.properties");
        //Configuration.Resource paraResource = new Configuration.Resource("big_dvd/efm-dvd-parameter.properties");
        Configuration.Resource paraResource = new Configuration.Resource("efm/efm-rating.properties");
        conf.addResource(resource);
        conf.addResource(paraResource);
        gridSearch grid = new gridSearch();
        EfmRecommender recommender = new EfmRecommender();
        //BPRRecommender recommender = new BPRRecommender();
        ModifyRecommenderJob job = new ModifyRecommenderJob(conf);
        job.setParameterSearch(grid);
        job.setRecommender(recommender);
        job.runJob();
    }
    public static void gridtest() throws ClassNotFoundException, LibrecException, IOException{
        Configuration conf = new Configuration();
        Configuration.Resource resource = new Configuration.Resource("grid-test.properties");
        conf.addResource(resource);
        parameter.gridSearch grid = new gridSearch();
        grid.setConf(conf);
        grid.setup();
        while (grid.schedule()) {
            System.out.println(Arrays.toString(grid.getParameterValues()));
        }
    }
}
