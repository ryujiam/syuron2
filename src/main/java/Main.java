import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.recommender.cf.ranking.BPRRecommender;
import net.librec.recommender.cf.rating.NMFRecommender;
import parameter.gridSearch;
import recommend.CMFRecommender;
import recommend.CMFSGDRecommender;
import recommend.EfmRecommender;
import recommend.EfmSGD;
import validation.validArffDataModel;
import validation.validKCVDataSplitter;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        //testRecommender();
        testValidRecommender();
        //gridtest();

    }

    public static void testRecommender() throws ClassNotFoundException, LibrecException, IOException {
        Configuration conf = new Configuration();
        //Configuration.Resource resource = new Configuration.Resource("efm-test.properties");
        //Configuration.Resource paraResource = new Configuration.Resource("big_dvd/efm-dvd-parameter.properties");
        Configuration.Resource paraResource = new Configuration.Resource("mf/bpr.properties");
        //Configuration.Resource paraResource = new Configuration.Resource("cmf/cmf-rating.properties");
        //conf.addResource(resource);
        conf.addResource(paraResource);
        gridSearch grid = new gridSearch();
        //CMFRecommender recommender = new CMFRecommender();
        //EfmRecommender recommender = new EfmRecommender();
        //NMFRecommender recommender = new NMFRecommender();
        BPRRecommender recommender = new BPRRecommender();
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

    public static void testValidRecommender() throws ClassNotFoundException, LibrecException, IOException {
        Configuration conf = new Configuration();
        //Configuration.Resource paraResource = new Configuration.Resource("cmf/cmf-sgd-rating.properties");
        //Configuration.Resource paraResource = new Configuration.Resource("cmf/cmf-sgd-rating.properties");
        Configuration.Resource paraResource = new Configuration.Resource("efm/efm-sim-ranking.properties");
        conf.addResource(paraResource);
        gridSearch grid = new gridSearch();
        //CMFRecommender recommender = new CMFRecommender();
        //CMFSGDRecommender recommender = new CMFSGDRecommender();
        EfmRecommender recommender = new EfmRecommender();
        //EfmSGD recommender = new EfmSGD();
        validKCVDataSplitter dataSplitter = new validKCVDataSplitter(conf);
        validArffDataModel dataModel = new validArffDataModel(conf);
        dataModel.setDatasplitter(dataSplitter);
        ModifyRecommenderJob job = new ModifyRecommenderJob(conf);
        job.setParameterSearch(grid);
        job.setAlterDataModel(dataModel);
        job.setRecommender(recommender);
        job.runJob();


    }
}
