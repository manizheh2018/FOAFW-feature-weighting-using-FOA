/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package foafw;

/**
 *
 * @author Ghaemi
 * Refer to the related paper to best understand how to change 
 * to the parameter values according to your dataset- 
 * paper- "Ghaemi, Manizheh, and Mohammad-Reza Feizi-Derakhshi. "Forest optimization algorithm." Expert Systems with Applications 41, no. 15 (2014): 6676-6687."
 */
import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;

public class global {
    public static int NumSeeds;
    public static int dim;
    public static int live;
    public static int arelimit;
    public static int transferRate;
    public static int MAxTime;
    public static LinkedList<double[]> trees;
    public static LinkedList<double[]> candidate;
    public static int knear;
    public static int LSC;
    public static int GSC;
    public void init()
    {
        NumSeeds=30;
        MAxTime=1000;//0;
        dim=14;// change this to the size of the dataset (#features)
        live=2;// Maximum lifetime of a tree
        arelimit=30;//The number of population members
        transferRate=5;
        knear=1; //the value of k in K nearest neighbor classification
        LSC=3;//Number of local seeding - change this according to the number of features
        GSC=6;//number of global seeding- change this according to the number of features
        trees=new LinkedList<double[]>();
        candidate=new LinkedList<double[]>();

    }

}
