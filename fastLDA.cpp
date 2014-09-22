/* Implementation of fast parallel algorithm for LDA : SCVB0
Authors : Sayan Ghosh, Atamananda Persaud
Important : Read usage instructions before building and running (given in README file)
*/

#include<math.h>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<iostream>
#include<fstream>
#include<string>
#include<time.h>
using namespace std;

// Initialize number of documents, topics and words in vocabulary
unsigned int W,D,K;

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        printf("Usage: ./fastLDA inputfile num_iterations num_topics [optional:minibatch size]\n");
        return 1;
    }

    // Initlialize expected topic counts per document
    float **N_theta;
    // Dynamically
    float **N_phi;
    float *N_z;
    float rho_theta=0;
    float rho_phi=0;
    float *gamma_ij;
    float **phi;
    float **theta;
    float *perplexities;
    int cnt;
    // Initlalize dirichlet prior parameters
    float alpha,eta;
    float M; // Number of documents in each minibatch
    int j,Cj=0,i,k,w;
    double norm_sum=0;
    int batch_idx=0;
    int C=0;
    //int *C_j;
    int MAXITER;
    int iter=0;
    unsigned int i_cnt=0;
    int NNZ;
    float perplexityval,innerval;
 
    int m_aj;     
    
    M=100; //343 works for KOS and only for KOS
    eta=0.01;// was 0.01
    alpha=0.1;

      
    MAXITER = atoi(argv[2]);
    K = atoi(argv[3]);

    printf("Input file: %s\n",argv[1]);
    printf("Number of iterations: %d\n",MAXITER);
    printf("Number of topics: %d\n",K);
    printf("Minibatch size: %f\n",M);
    printf("alpha:  %f\n",alpha);
    printf("eta:  %f\n",eta);

    // Read the file and store it in DATA
    FILE* fptr;
    unsigned int docnum,wnum;
    unsigned char countnum;

    fptr=fopen(argv[1],"rt");

    fscanf(fptr, "%d\n", &D);
    fscanf(fptr, "%d\n", &W);
    fscanf(fptr, "%d\n", &NNZ);

    printf("Number of documents: %d\n",D);
    printf("Vocabulary size: %d\n",W);

 
    // Dynamically allocate phi
    phi=new float*[W];

    for (w=0;w<W;w++)
    {
        phi[w]=new float[K];
    }

    printf("allocated phi\n");

    // Dynamically allocate theta

    theta=new float*[D];

    for (i=0;i<D;i++)
    {
        theta[i]=new float[K];
    }

    printf("allocated theta\n");

    printf("allocated data\n");

    //if user also specified a minibatch size
    if(argc == 5 || argc==6)
    {
        M = atof(argv[4]);
    }

    vector <vector <int> > corpus;
    vector<int> corpus_size(D,0);
    corpus.resize(D);
    //vector <vector <int> > corpus_expanded;
    //corpus_expanded.resize(D);

    while(!feof(fptr))
    {
        fscanf(fptr,"%d %d %hhu\n",&docnum,&wnum,&countnum);
       
            corpus[docnum-1].push_back(wnum-1);
            corpus[docnum-1].push_back(countnum);
            
            corpus_size[docnum-1]+=countnum;
        /*
        for(i=0;i<countnum;i++)
        {
	    corpus_expanded[docnum-1].push_back(wnum-1);
        }
        */
     
    }
    fclose(fptr);

   

    // Initialize phi_est and all other arrays
 
    N_phi=new float*[W];

    for (i=0;i<W;i++)
    {
        
        N_phi[i]=new float[K];
    }

    
    for (i=0;i<W;i++)
    {
        for (k=0;k<K;k++)
        {
            
            N_phi[i][k]=rand()%10;
        }
    }

    // Initialize n_z and n_z_est and other arrays
    N_z=new float[K];
    //N_z_est=new float[K];
    
    for (k=0;k<K;k++)
    {
        N_z[k]=0;
        
    }

    
    for (k=0;k<K;k++)
    {
        for (w=0;w<W;w++)
        {
            N_z[k]+=N_phi[w][k];
        }
    }

    perplexities = new float[MAXITER];
    for (i=0;i<MAXITER;i++)
    {
      perplexities[i]=0;
    }

    N_theta=new float*[D];
   
    for (i=0;i<D;i++)
    {
        N_theta[i]=new float[K];
    }
    
    for (i=0;i<D;i++)
    {
        for (k=0;k<K;k++)
        {
            N_theta[i][k]=rand()%10;
        }
    }

    for(j=0;j<D;j++)
    {
        C += corpus_size[j];
    }

    printf("Number of words in corpus: %d\n", C);

    //gamma_ij=new float[K];

    int firstdoc = 0;
    int lastdoc = 0;
    int DM = D/M;

    for (iter=0;iter<MAXITER;iter++)
    {
        // Decide rho_phi and rho_theta
        rho_phi=10/pow((1000+iter),0.9);
        rho_theta=1/pow((10+iter),0.9);

	#pragma omp parallel private(batch_idx,j,k,norm_sum,i,w,firstdoc,lastdoc)
        {
        float *gamma_ij = new float[K];
        float *N_z_est = new float[K];
        float **N_phi_est = new float *[W];
        for (k=0;k<K;k++)
        {
            gamma_ij[k]=0;
            N_z_est[k]=0;
        }
        for (i=0;i<W;i++)
        {
            N_phi_est[i]=new float[K];
            for (k=0;k<K;k++)
            {
                N_phi_est[i][k]=0;
            }
        }

        #pragma omp for
        for (batch_idx=0;batch_idx<DM;batch_idx++)
        {
            firstdoc = batch_idx*M;
            lastdoc = (batch_idx+1)*M;

	    for (j=firstdoc;j<lastdoc;j++)
            {	        
                
                
                // Store size of corpus in Cj
                Cj=corpus_size[j];
               
               
                for (i=0;i<(corpus[j].size()/2);i++) // indexing is very different here!
                {
                    
                    int w_aj=corpus[j][2*i];
                    int m_aj=corpus[j][(2*i)+1];
                    // Update gamma_ij and N_theta
                    float norm_sum=0;
                    
     
                    for (k=0;k<K;k++)
                    {
                        gamma_ij[k]=(N_phi[w_aj][k]+eta)*(N_theta[j][k]+alpha)/(N_z[k]+(eta*W));
                        norm_sum+=gamma_ij[k];
                    }

                   
                    for (k=0;k<K;k++)
                    {
                        gamma_ij[k]=gamma_ij[k]/norm_sum;
                    }

                
                    for (k=0;k<K;k++)
                    {

                      N_theta[j][k]=(pow((1-rho_theta),m_aj)*N_theta[j][k])+((1-pow((1-rho_theta),m_aj))*Cj*gamma_ij[k]);
//                      	      
                    }

                }

               
               
              
                // Iteration of the main loop
             
                for (i=0;i<(corpus[j].size()/2);i++) // indexing is very different here!
                {

                    int w_aj=corpus[j][2*i];
                    int m_aj=corpus[j][(2*i)+1];
                    norm_sum=0;
                    for (k=0;k<K;k++)
                    {
                        gamma_ij[k]=(N_phi[w_aj][k]+eta)*(N_theta[j][k]+alpha)/(N_z[k]+(eta*W));
                        norm_sum+=gamma_ij[k];
                    }

                    //# pragma omp parallel for
                    for (k=0;k<K;k++)
                    {
                        gamma_ij[k]=gamma_ij[k]/norm_sum;
                    }

                    // Update N_theta estimates
                    for (k=0;k<K;k++)
                    {
                        N_theta[j][k]=(pow((1-rho_theta),m_aj)*N_theta[j][k])+((1-pow((1-rho_theta),m_aj))*Cj*gamma_ij[k]);
                        N_phi_est[w_aj][k]=N_phi_est[w_aj][k]+(C*gamma_ij[k]/M);

                        N_z_est[k]=N_z_est[k]+(C*gamma_ij[k]/M);
                    }
                }
               
            } // End of j

          
            // Update the estimates matrix
            for (k=0;k<K;k++)
            {
                for (w=0;w<W;w++)
                {
                    N_phi[w][k]=(1-rho_phi)*N_phi[w][k]+rho_phi*N_phi_est[w][k];
                }
                #pragma omp atomic
                N_z[k]*=(1-rho_phi);
                #pragma omp atomic
                N_z[k]+=rho_phi*N_z_est[k];
               
            }
             
        } // End of batch_idx
        
/*--------------------------------------COMPUTATION OF THETA AND PHI---------------------------------*/
        // Compute phi
        #pragma omp for
        for (k=0;k<K;k++)
        {
            norm_sum=0;
            for (w=0;w<W;w++)
            {
                N_phi[w][k]+=eta;
                norm_sum+=N_phi[w][k];
            }
            for (w=0;w<W;w++)
            {
                phi[w][k]=(float)N_phi[w][k]/norm_sum;
            }
        }

        // Compute theta
        #pragma omp for
        for (i=0;i<D;i++)
        {
            norm_sum=0;
            for (k=0;k<K;k++)
            {
                N_theta[i][k]+=alpha;
                norm_sum+=N_theta[i][k];
            }
            for (k=0;k<K;k++)
            {
                theta[i][k]=(float)N_theta[i][k]/norm_sum;
            }
        }

        delete [] gamma_ij;
        delete [] N_z_est;

        for(i=0;i<W;i++)
        {
            delete [] N_phi_est[i];
        }

        delete [] N_phi_est;

        }

/******************************************CALCULATE PERPLEXITY**************************************************
        // Calculate the perplexity here
        // Compute posterior means here
        // Iterate over the corpus here
        perplexityval=0;
        #pragma omp parallel for private(j,i,k) reduction(+:innerval) reduction(+:perplexityval)
        for (j=0;j<D;j++)
	{
		for (i=0;i<corpus_expanded[j].size();i++)
		{
                        innerval=0;
			for (k=0;k<K;k++)
			{
				innerval+=(theta[j][k]*phi[corpus_expanded[j][i]][k]);
			}
			perplexityval+=(log(innerval)/log(2));
		}
	}
	printf("%d,%f\n",iter,pow(2,-perplexityval/C));
        perplexities[iter] = pow(2,-perplexityval/C);
        
        pfile << iter+1 << "," << perplexities[iter] << endl;
        pfile.flush();
*/
    } // End of iter
 
    //write doctopics file
    ofstream dtfile;
    dtfile.open("doctopic.txt");
    for (i=0;i<D;i++)
    {
        for (k=0;k<K;k++)
        {
            dtfile << theta[i][k] << ",";
        }
        dtfile << endl;
    }
    dtfile.close();

    //compute the top 100 words for each topic
    int** topwords;
    float** maxval;
    topwords = new int*[K];
    maxval=new float*[K];
    for (k=0;k<K;k++)
    {
        topwords[k] = new int[100];
        maxval[k]=new float[100];
    }

    for (k=0;k<K;k++)
    {
        for (i=0;i<100;i++)
	{
	    float max = -1;
	    int max_idx = -1;
	    for (w=0;w<W;w++)
	    {
	        if (phi[w][k] > max)
		{
		    max = phi[w][k];
		    max_idx = w;
		}
	    }
	    phi[max_idx][k] = 0;
	    topwords[k][i] = max_idx;
            maxval[k][i]=max;
	}   
    }
    
    //printf("1\n");
    string *dict;
    dict = new string[W];
    /*fptr=fopen(argv[5],"rt");
    printf("after fptr\n");
    char word;*/
    
    // while(!feof(fptr))
    // {
        
    //     fscanf(fptr,"%s\n",&word);
    // 	dict[w]=word;
    // 	w++;
    // 	printf("%d",w);
    // }
    // fclose(fptr);

    //write topics file
    ofstream tfile;
    tfile.open("topics.txt");
    for (k=0;k<K;k++)
    {
        for (w=0;w<100;w++)
        {
            tfile << topwords[k][w] << ":" << maxval[k][w] << ",";

            
        }
        tfile << endl ;

        
    }
    tfile.close();

      
    return(0);

} // End of main
