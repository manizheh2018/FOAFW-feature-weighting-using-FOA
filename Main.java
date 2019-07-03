/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
/*Refer to the related paper to best understand how to change 
 * to the parameter values according to your dataset- 
 * paper- "Ghaemi, Manizheh, and Mohammad-Reza Feizi-Derakhshi. "Forest optimization algorithm." Expert Systems with Applications 41, no. 15 (2014): 6676-6687."
 */
package foafw;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Ghaemi
 */
public class Main {

    
    public Instances data, train, test,temptrain, temptest;
    String trainAd;
    public static boolean first=true;
    public static double[] best=new double[global.dim+2];
    public static boolean istest=false;
    public static int notchange=0;

    //******************* reading the train and test set ************
     public  void readArffFiles(String dataAdress){
        try {
            double[] contain;
            double[] mi=new double[global.dim];
            double[] ma=new double[global.dim];
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataAdress);
            data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            // ***********************normalization*********************//
            double[][] minmax=new double[data.numInstances()][data.numAttributes()];
            for (int i=0;i<data.numInstances();i++){
                contain=data.instance(i).toDoubleArray();
                System.arraycopy(contain, 0, minmax[i], 0, global.dim);
            }
            //finding the minimum values

            for (int i=0;i<minmax[0].length-1; i++ )
                {
                    mi[i]=Integer.MAX_VALUE;
                    for(int j=0;j<minmax.length;j++)
                        if ( minmax[j][i]<mi[i] )
                            mi[i]=minmax[j][i];
                }
            //finding the maximum values
            for (int i=0;i<minmax[0].length-1; i++ )
                {
                    ma[i]=Integer.MIN_VALUE;
                    for(int j=0;j<minmax.length; j++ )
                        if ( minmax[j][i]>ma[i] )
                            ma[i]=minmax[j][i];
                }
            for (int i=0;i<data.numInstances();i++){
                contain=data.instance(i).toDoubleArray();
                for (int j=0;j<global.dim-1;j++)
                    data.instance(i).setValue(j,Math.round((contain[j]-mi[j])/(ma[j]-mi[j]) * 100.0 ) / 100.0 );
            }
            // **************** specifing  70% train and 30% test data sets ******************
            data.randomize(new Random(1));
            double trainpercent = 70.0;
            int trainSize = (int) Math.round(data.numInstances() * trainpercent / 100);
             int testSize = data.numInstances() - trainSize;//-crossSize;
            train = new Instances(data, 0, trainSize);
            test = new Instances(data, trainSize, testSize);
            if (train.classIndex() == -1) {
                train.setClassIndex(train.numAttributes() - 1);
            }
            if (test.classIndex() == -1) {
                test.setClassIndex(test.numAttributes() - 1);
            }
            //System.out.print("data has"+data.numInstances()+"train has"+train.numInstances()+"cross has"+cross.numInstances()+"test has"+test.numInstances());
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    //************** returns the fitness value ********************
    public  double fitness(double v[]){
        try{
            double accuracy=0;
            addweight(v);// multiplies weights to values
            IBk ibk= new IBk(global.knear);
            Evaluation eval=new Evaluation(temptrain);
            if (!istest){
                ibk.buildClassifier(temptrain);
                eval.crossValidateModel(ibk, temptrain,10, new Random(1));
                accuracy=(eval.correct()/temptrain.numInstances())*100;
            }
            else{
                ibk.buildClassifier(temptrain);
                eval.evaluateModel(ibk, temptest);
                accuracy=(eval.correct()/temptest.numInstances())*100;
                System.out.print(eval.toSummaryString("\nResults\n\n", false));///false
            }
            //if (accuracy>60)
              // System.out.print("accuracy is bigger than 60 : "+accuracy+"\n");
            return accuracy;
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            return -1;
        }
    }
    //************** adding weights to the features*****************
    public  void addweight(double v[]){
         try {
             double[] contain=new double[global.dim];
             /* train data*/
             temptrain= new Instances(train);
                         if (temptrain.classIndex() == -1) {
                             temptrain.setClassIndex(temptrain.numAttributes() - 1);
                         }
             for (int i=0;i<temptrain.numInstances();i++){
                 contain=train.instance(i).toDoubleArray();
                 for (int j=0;j<global.dim-1;j++)
                     temptrain.instance(i).setValue(j,v[j]*contain[j]);
             }
              
             /* test data*/
             temptest= new Instances(test);
                         if (temptest.classIndex() == -1) {
                             temptest.setClassIndex(temptest.numAttributes() - 1);
                         }
             for (int i=0;i<temptest.numInstances();i++){
                 contain=test.instance(i).toDoubleArray();
                 for (int j=0;j<global.dim-1;j++)
                     temptest.instance(i).setValue(j,v[j]*contain[j]);
             }

         } catch (Exception ex) {
             Logger.getLogger( Main.class.getName()).log(Level.SEVERE, null, ex);
         }
     }
    //*************** initializing initial forest with random trees ********
    public void initailization(){
        /** including class attribute **/

        //readArffFiles("vehicle.arff"); //#attr=19
        //readArffFiles("srbct.arff"); //#attr=2309 ib3 94.74
        readArffFiles("cleveland.arff");//#attr=14 64.35 j48// ib1 61.05/ib3 62.37/ib5 63.69/ib7 63.03/ib9 63.69/ib11 63.69
        //readArffFiles("liverbupa.arff"); //#attr=7 ib1 68.40/ ib3 68.98/ib5 68.11 ib7 68.40/ ib9 68.40/ib11 70.14
        //readArffFiles("glass.arff"); //#attr=10 ib3 75.23/ib1 79.43 79.69/ib5 74.76/ib5 73.83/ ib7 73.83/ib9 71.96 /ib11  71.02
        //readArffFiles("heart-statlog.arff"); //#attr=14 //86.29 ibk k=5// nb 87.03//j48 85.55
        //readArffFiles("pima.arff"); //#attr=9 //ib9 75.27/75.09 j48/ib1 72.20/ib3 74.54/ib5 75.99/ib7 75.27
        //readArffFiles("sonar.arff"); //#attr=61/ ib1 94.71 92.30/
        //readArffFiles("ionosphere.arff"); //#attr=35 ib7 90.59/ib3 93.44
        //readArffFiles("hepatitis.arff"); //#attr=20
        //readArffFiles("segment-challenge.arff"); //#attr=20
        //readArffFiles("liverbupa.arff");//#fe=7
        double[][] tree= new double[global.NumSeeds][global.dim+2];
        double[] pass=new double[global.dim];
        double[][] sortedTrees;
        int di=global.dim;
        int lives=0;
        for (int i=0;i<global.NumSeeds;i++){
            for (int j=0;j<global.dim;j++)
                tree[i][j]=Math.round( Math.random() * 10.0 ) / 10.0;//Math.round(Math.random());
            System.arraycopy(tree[i], 0, pass, 0, global.dim);
            tree[i][global.dim]=fitness(pass);
            //System.out.print(tree[i][global.dim]+"  ");
            tree[i][di+1]=0;
            global.trees.add(tree[i]);
        }
    }
    //*************** form the neighbours of the seeds with 0 live ***********
    public void localSeeding(){

        LinkedList<double[]> newtrees=new LinkedList<double[]> ();
        int di=global.dim ;
        int [][] jk=new int[global.dim-1][1];
        for (int op=0;op<global.dim-1;op++)
            jk[op][0]=op;
        List<int[]> lo = new ArrayList<int[]>(Arrays.asList(jk));

        /*System.out.print("live of trees before local seeding are:\n");
        for (double[] s:global.trees){
            System.out.print(s[global.dim+1]+"   ");
        }*/
        for (double[] s:global.trees){
            if (s[global.dim+1]==0){              
                int[] moveindex=new int[global.LSC];
                Collections.shuffle(lo);
                jk = lo.toArray(new int[][]{});
                for(int y=0;y<global.LSC;y++)
                    moveindex[y]=jk[y][0];
                for (int m:moveindex){
                    double[] trans=new double[global.dim+2];
                    System.arraycopy(s, 0, trans, 0, global.dim);
                    double ad=.1;//(Math.random() )/10;//(Math.round( Math.random() * 10.0 ))/100;
                    if(trans[m]+ad<=1){
                        trans[m]=trans[m]+ad;
                        trans[global.dim]=fitness(trans);
                        trans[di+1]=0;
                        newtrees.add(trans);
                        
                    }
                    // second neigbour
                    trans=new double[global.dim+2];
                    System.arraycopy(s, 0, trans, 0, global.dim);
                    if(trans[m]-ad>=0){
                        trans[m]=trans[m]-ad;
                        trans[global.dim]=fitness(trans);
                        trans[di+1]=0;
                        newtrees.add(trans);
                    }
                }
            }
            s[di+1]=s[di+1]+1;
        }
       // System.out.print("the size of global trees before adding locals "+global.trees.size()+" \n");
        for (double[] tt: newtrees)
            global.trees.addLast(tt);
      newtrees.clear();

    }
    //*******************************control forest*************
    public void controlForest(){
        int index=0;
        int di=global.dim;
        double[] cf=new double[di+2];
        LinkedList<double[]> temp=new LinkedList<double[]> ();
        double[][] sortedTrees;
        
        for(double[] tree:global.trees){
            //System.out.print("live is"+tree[global.dim+1]+" \n ");
            if (tree[di+1]<=global.live)
                temp.add(tree);
            else
                global.candidate.add(tree);
        }
        if (temp.size()>global.arelimit)//t
        {
            sortedTrees = temp.toArray(new double[][]{});
            Arrays.sort(sortedTrees, new Comparator<double[]>() {
	    public int compare(double[] a1, double[] a2) {
                return Double.valueOf(a1[global.dim]).compareTo(Double.valueOf(a2[global.dim]));
            }
            });
            temp.clear();
            for (int q=0;q<sortedTrees.length;q++)
            {
                temp.add(sortedTrees[q]);
            }
            Collections.reverse(temp);
            sortedTrees=temp.toArray(new double[][]{});
            int siz=temp.size();
            for(int y=0;y<(siz-global.arelimit);y++){
               global.candidate.addLast(sortedTrees[y]);
               temp.removeLast();
           }
        }
        //System.out.print(" \n the candid size is  "+global.candidate.size());
        global.trees.clear();
        for (double[] tree:temp)
            global.trees.add(tree);
        //System.out.print("the size of global trees after deleting candid  "+global.trees.size()+" \n");
    }
    //********************global seeding *****************
    public void globalSeeding(){
        
        LinkedList<double[]> newtrees=new LinkedList<double[]> ();
        int revosize,extra;
        double [] pass=new double[global.dim+2];
        int [][] jk=new int[global.dim-1][1];
        int[] moveindex=new int[global.GSC];
        for (int op=0;op<global.dim-1;op++)
                    jk[op][0]=op;
        List<int[]> lo = new ArrayList<int[]>(Arrays.asList(jk));
        Collections.shuffle(lo);
        jk = lo.toArray(new int[][]{});
        for(int y=0;y<global.GSC;y++)
            moveindex[y]=jk[y][0];
        revosize=(global.candidate.size()>global.transferRate?Math.round((global.candidate.size()*global.transferRate)/100):global.candidate.size());
        extra=(Math.abs(revosize-global.candidate.size()));
        for(int k=0;k<extra;k++)
            global.candidate.removeLast();
        //System.out.print("the size of candidate is  "+global.candidate.size()+" \n");
        for (double[] inn:global.candidate){
            System.arraycopy(inn, 0, pass, 0, global.dim);
            for (int m:moveindex)
            {
                pass[m]=Math.round( Math.random() * 10.0 ) / 10.0;
            }
            pass[global.dim]=fitness(pass);
            pass[global.dim+1]=0;
            global.trees.add(pass);
        }
        
        //System.out.print("the size of global trees after global "+global.trees.size()+"  and candiate was \n"+global.candidate.size()+"\n");
        global.candidate.clear();
    }
    //*************************
    public static  int findbestindex(){
        double[][] sortedTrees;
        int bestindex=100000;
        double bestone=0;
        int di=global.dim;
        double[] nm=new double[global.dim+2];
        LinkedList<double[]> temp=new LinkedList<double[]> ();
        sortedTrees = global.trees.toArray(new double[][]{});
        for(int y=0;y<sortedTrees.length;y++)
            if (sortedTrees[y][global.dim] > bestone){
                bestone=sortedTrees[y][global.dim];
                bestindex=y;
            }
        //first=false;
        System.out.print(" The best classification accuracy found is  "+sortedTrees[bestindex][global.dim]);//+"\n with index  "+bestindex+"\n");
        return bestindex;
    }
    //*********** update best tree **************
    public void updateBest()
    {
        int bestindex;
        int di=global.dim;
        double[] temp;
        bestindex=findbestindex();
        //System.out.print("\n best index is \n"+bestindex+" \n");
        temp=global.trees.get(bestindex);
        global.trees.remove(bestindex);
        temp[di+1]=0;
        global.trees.addFirst(temp);
        if ((temp[global.dim]==best[global.dim]))//bestindex==0&&
        {
            notchange=notchange+1;
        }
        else if (temp[global.dim]>best[global.dim])
        {
            notchange=1;
            System.arraycopy(temp, 0, best, 0, global.dim+2);
        }
        else
            System.out.print("\n problem !!!!!!!!!! \n");

        //System.out.print("\n notchange is \n"+notchange+" \n");

    }

    public static void main(String[] args) {
        global g=new global();
        g.init();
        int bestindex=0;
        double accuracy=0;
        Main fs=new Main();
        fs.initailization();
        bestindex=(int)Math.round(Math.random()*global.trees.size());
        best=global.trees.get(bestindex);
        
        for (int r=0;r<global.MAxTime && notchange<50 ;r++)//
        {
            System.out.print("\n The iteration number is:"+r+"\n");
            fs.localSeeding();
            fs.controlForest();
            fs.globalSeeding();
            fs.updateBest();
        }
        accuracy=fs.fitness(best);
        for (int y=0;y<best.length;y++)
            System.out.print(best[y]+"  ");

        System.out.print("\n The classification accuracy is "+accuracy);
        istest=true;
        accuracy=fs.fitness(best);
        for (int y=0;y<best.length;y++)
            System.out.print(best[y]+"  ");
    }
}
//
