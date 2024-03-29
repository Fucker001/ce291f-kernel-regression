package kernel;

/*
  Copyright: Copyright (c) 1998-2009 MOSEK ApS, Denmark. All rights reserved.

  File:      example_lines.java

  Purpose: This example is not meant to be distributed. It
           demonstrates some functionality used in the manual.
           It is means to compile _only_! It will not run!
*/


public class example_lines
{
    public static void main (String[] args)
    {
        mosek.Env
            env = null;
        mosek.Task
            task = null;
        int
            numvar = 0;
        try
            {
                env  = new mosek.Env ();
                env.init ();
                task = new mosek.Task (env, 0,0);
                {
                  double[] c = new double[numvar];
                  task.getc(c);
                }
                {
                  double[] upper_bound     = new double[8];
                  double[] lower_bound     = new double[8];
                  mosek.Env.boundkey bound_key[] 
                                           = new mosek.Env.boundkey[8];
                  task.getboundslice(mosek.Env.accmode.con, 2,10,
                                     bound_key,lower_bound,upper_bound);
                }
                {
                  int[]  bound_index   = { 1, 6, 3, 9 };
                  mosek.Env.boundkey[] bound_key 
                                       = { mosek.Env.boundkey.fr,
                                           mosek.Env.boundkey.lo,
                                           mosek.Env.boundkey.up,
                                           mosek.Env.boundkey.fx };
                  double[] lower_bound = { 0.0, -10.0, 0.0, 5.0 };
                  double[] upper_bound = { 0.0,   0.0, 6.0, 5.0 };
                  task.putboundlist(mosek.Env.accmode.con, bound_index,
                                    bound_key,lower_bound,upper_bound);
                }
                {
                  int[]    subi = {   1,   3,   5 };
                  int[]    subj = {   2,   3,   4 };
                  double[] cof  = { 1.1, 4.3, 0.2 };
                  task.putaijlist(subi,subj,cof);
                }


                {
                  int[] rowsub = { 0, 1, 2, 3 };
                  int[] ptrb   = { 0, 3, 5, 7 };
                  int[] ptre   = { 3, 5, 7, 8 };
                  int[] sub    = { 0, 2, 3, 1, 4, 0, 3, 2 };
                  double[] cof = { 1.1, 1.3, 1.4, 2.2, 2.5, 3.1, 3.4, 4.4 };
                  
                  task.putaveclist (mosek.Env.accmode.con,
                                    rowsub,ptrb,ptre,
                                    sub,cof);
                }

            }
        catch (mosek.ArrayLengthException e)
            {
                System.out.println ("Error: An array was too short");
                System.out.println (e.toString ());
            }
        catch (mosek.Exception e)
            /* Catch both mosek.Error and mosek.Warning */
            {
                System.out.println ("An error or warning was encountered");
                System.out.println (e.getMessage ());
            }
        
        if (task != null) task.dispose ();
        if (env  != null)  env.dispose ();
    }
}
