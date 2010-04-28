// Traffic analysis

//on Franklin, requires "module load acml"

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <time.h>
#include <acml.h>
#include <string>
#include <time.h>
#include <sys/time.h>


//Constants for data sending
#define L1RESULT 0
//for blocking in calculations
#define BLOCKSIZE 32

#define EPSILON 0.01

#define RHO 0.01

//
//  timer
//
double read_timer( ){
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//Prints iDim x jDim matrix data
//Comes in column major, print in row
void printMat(double *matrix,unsigned int iDim,unsigned int jDim){
	for(unsigned int i=0;i<iDim;i++){
		for(unsigned int j=0;j<jDim;j++){
			printf("%f ",matrix[i+j*iDim]);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

void calcSVD(double *in,double*out,unsigned int dim){
    double simulation_time = read_timer( );
	
	//Calculates U*S^.5 where in = U*S*V^T, stores in out
	double *S = (double*)calloc(dim,sizeof(double));
	double *U = (double*)calloc(dim*dim,sizeof(double));
	double *Vt = (double*)calloc(dim*dim,sizeof(double));
	int info=0;
	
	//DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK,LWORK, IWORK, INFO )
	dgesdd('A',dim,dim,in,dim,S,U,dim,Vt,dim,&info);
	
	/*
	
	//ORIGINAL
	//Multiply U*S^.5, store in out
	for(unsigned int i=0;i<dim;i++){
		for(unsigned int j=0;j<dim;j++){
			out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
		}
	}
	*/
	
	//Blocked
	int blockSize = min(BLOCKSIZE,(int)dim);
	for(int y=0;y<((int)ceil(1.0*dim/blockSize));y++){
		for(int x=0;x<ceil(1.0*dim/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<dim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<dim;j++){
					out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
				}
			}
		}	
	}
	// divide by RHO
	for(int i=0;i<dim*dim;i++){
		out[i] = out[i] / RHO;
	}
		
	//out now holds U*(S^.5)
	// ROW MAJOR i+j*dim
	// column major: j+i*dim
	
	//free calloc'ed space
	free(S);
	free(U);
	free(Vt);
	return;
}

void calcPolyKernel(double *in, double *out, unsigned int iDim, unsigned int jDim, double c1,double c2, double d){
	double *xi, *xj, xd[1], result;
	unsigned int maxDim = max(iDim,jDim);
	double *intermed = (double*)calloc(maxDim*maxDim,sizeof(double));
	double *intermed2 = (double*)calloc(maxDim*maxDim,sizeof(double));
	if (intermed==NULL||intermed2==NULL){
		printf("intermed calloc fail\n");
		return;
	}
	//get in^T to form column-major...
	/*
		//original
		
	for(unsigned int i=0;i<iDim;i++){
		for(unsigned int j=0;j<jDim;j++){
			intermed2[i+j*iDim] = in[j+i*jDim];
		}
	}
	*/
	
	//Blocked
	int blockSize = min(BLOCKSIZE,(int)min(jDim,iDim));
	for(int y=0;y<((int)ceil(1.0*iDim/blockSize));y++){
		for(int x=0;x<ceil(1.0*jDim/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<iDim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<jDim;j++){
					intermed2[i+j*iDim] = in[j+i*jDim];
					//out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
				}
			}
		}			
	}
	
	//Form K[i,j]_(N,N)
	for(unsigned int i=0;i<jDim;i++){
		//get ith column
		xi = &(intermed2[i*iDim]);
		for(unsigned int j=0;j<jDim;j++){
			//get jth column
			xj = &(intermed2[j*iDim]);
			xd[0]=0;
			dgemm( 'T','N', 1,1,iDim, c2, xi,iDim, xj,iDim, 1, xd,1 );
			result = c1+xd[0];
			intermed[j+i*jDim] = pow(result,d);
		}
	}
	
	//Normalization
	double trace = 0;
	for(unsigned int i=0;i<maxDim;i++){
		trace += intermed[i+i*maxDim];	
	}
	double toD = 1/trace;
	//Scale by normalization factor
	dscal(maxDim*maxDim, toD, intermed,1);
	
	//compute SVD, store in out
	calcSVD(intermed,out,maxDim);
	
	//free calloc'ed space
	free(intermed);
	free(intermed2);
	return;
}
void calcGaussKernel(double *in,double *out,unsigned int iDim, unsigned int jDim,double sigma){
    double simulation_time = read_timer( );
	
	double *xi,*xj;
	double xd[iDim];
	double sum, result;
	unsigned int maxDim = max(iDim,jDim);
	double *intermed = (double*)calloc(maxDim*maxDim,sizeof(double));
	double *intermed2 = (double*)calloc(maxDim*maxDim,sizeof(double));
	
	/* //original
	//put array into column major
	for(unsigned int i=0;i<iDim;i++){
		for(unsigned int j=0;j<jDim;j++){
			intermed2[i+j*iDim] = in[j+i*jDim];
		}
	}
	*/
	//Blocked
	int blockSize = min(BLOCKSIZE,(int)min(jDim,iDim));
	for(int y=0;y<((int)ceil(1.0*iDim/blockSize));y++){
		for(int x=0;x<ceil(1.0*jDim/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<iDim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<jDim;j++){
					intermed2[i+j*iDim] = in[j+i*jDim];
					//out[j+i*dim] = sqrt(S[i])*U[j+i*dim];
				}
			}
		}			
	}
	//array in column major
	for(unsigned int i=0;i<jDim;i++){
		//get ith column
		xi = &(intermed2[i*iDim]);
		for(unsigned int j=0;j<jDim;j++){
			//get jth column
			xj = &(intermed2[j*iDim]);
			for(unsigned int q=0;q<iDim;q++){
				//for each item, find difference
				xd[q] = xi[q]-xj[q];
			}
			//sum items in column + norm it
			sum = 0;
			for(unsigned int q=0;q<iDim;q++){
				sum += pow(abs(xd[q]),2);
			}
			//square root sum
			result = sqrt(sum);
			//square to normalize
			intermed[j+i*jDim] = pow(result,2);
		}
	}
	
	//need exponential... of intermed
	for(unsigned int i=0;i<maxDim*maxDim;i++){
		intermed[i] = exp(-1*(intermed[i]/(2*pow(sigma,2))));
	}
	
	//Normalization
	double trace = 0;
	for(unsigned int i=0;i<maxDim;i++){
		trace += intermed[i+i*maxDim];	
	}
	double toD = 1/trace;
	dscal(maxDim*maxDim, toD, intermed,1);
	
	//compute SVD, store in out
	calcSVD(intermed,out,maxDim);
	
	free(intermed);
	free(intermed2);
	return;
}

void calcLinKernel(double *in,double *out,unsigned int iDim,unsigned int jDim){
	unsigned int maxDim = max(iDim,jDim);
	unsigned int minDim = min(iDim,jDim);
	double *intermed = (double*)calloc(maxDim*maxDim,sizeof(double));
	if (intermed==NULL){
		printf("intermed calloc fail\n");
		return;
	}
	//multiply X'*X, X already in row major, X' is column	
	dgemm('N', 'T', maxDim, maxDim, minDim, 1, in, maxDim, in, maxDim, 1, intermed, maxDim);
	
	//Normalization
	double trace = 0;
	for(unsigned int i=0;i<maxDim;i++){
		trace += intermed[i+i*maxDim];	
	}
	double toD = 1/trace;
	//Scale by normalization factor
	dscal(maxDim*maxDim, toD, intermed,1);
	
	// compute SVD, store in out
	calcSVD(intermed,out,maxDim);
	
	//free calloc'ed space
	free(intermed);
	return;

}
//Makes sure option is found
int find_option( int argc, char **argv, const char *option ){
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}
//Grabs data from commandline args
char *read_string( int argc, char **argv, const char *option, char *default_value ){
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}
//grab int from command line...
int read_int( int argc, char **argv, const char *option, int default_value ){
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}
//Prints correct usage info
void printUsage(){
	printf( "Options:\n" );
	printf( "-h to see this help\n" );
	printf( "-i <file> Input CSV file with input characteristics\n" );
	printf( "-o <file> Observed travel time CSV file \n" );
	printf( "-f <file> Kernel configuration\n" );
	printf( "-m <int> Max data points to use. ~1000\n" );
}
//Gets the number of values separated by delim (how many values in csv line)
unsigned int getNumIn(char *record, char *delim){
    char*p=strtok(record,delim);
    unsigned int fld=0;
    while(p){
		fld++;
		p=strtok('\0',delim);
	}		
	return fld;
}
//Parses csv line (record) using delim as separator. Stores each one in row arr
void parse( char *record, char *delim, double *arr,int actuallyAre,int numInputPoints){
    char *p=strtok(record,delim);
	double t;
	unsigned int fld=0;
	int offsetSelect = max(floor(actuallyAre / numInputPoints ),1.0);
	// printf("Offset: %d\n",offsetSelect);
	int c = 0;
    while(p){
		c = 0;
		t = atof(p);
        memcpy(&(arr[fld]),&t,sizeof(double));
		while (c < offsetSelect){
			p=strtok('\0',delim);
			c++;
		}
		fld++;
	}		
}
//Parses kernel.csv for kernel config...
/* format
G,sigma
L
P,c1,c2,d

*/
void parseKernel( char *record, char *delim, int *numG, int *numL, int *numP, double *sigma,double *c1,double *c2,double *d){
    char *p=strtok(record,delim);
	double t;
	// printf("__Found char: %c %s\n",p[0],p);
	if (p[0]=='G'){
		//gaussian input
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(sigma[*numG]),&t,sizeof(double));
		(*numG)++;
	}else if(p[0]=='L'){
		//Linear input
		(*numL)++;
	}else if(p[0]=='P'){
		//Poly input
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(c1[*numP]),&t,sizeof(double));
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(c2[*numP]),&t,sizeof(double));
		p=strtok('\0',delim);
		t = atof(p);
		memcpy(&(d[*numP]),&t,sizeof(double));
		(*numP)++;
	}else{
		printf("Invalid kernel csv\n");
		exit(1);
	}
	while(p){
		p=strtok('\0',delim);
	}
}
//only gets kernel counts so far
void parseKernelInit( char *record, char *delim, int *numG, int *numL, int *numP){
    char *p=strtok(record,delim);
	// printf("++Found char: %c %s\n",p[0],p);
	if (p[0]=='G'){
		//gaussian input
		(*numG)++;
	}else if(p[0]=='L'){
		//Linear input
		(*numL)++;
	}else if(p[0]=='P'){
		//Poly input
		(*numP)++;
	}else{
		printf("Invalid kernel csv\n");
		exit(1);
	}
	while(p){
		p=strtok('\0',delim);
	}
}
void getOTminusKtU(double *KtU,double *u, int Nt, int Np,double *rMajor, double *output, int Kdim ){
	// dgemv('N',Nt,Np,1.0,cMajor,Nt,u,1,1.0,KtU,1);
	for(int i=0;i<Nt;i++){
		for(int j=0;j<Np;j++){
			KtU[i] += rMajor[j+i*Kdim]*u[j];
		}
	}
	for(int i=0;i<Nt;i++){
		KtU[i] = output[i]-KtU[i];
	}
}
void getBeta(double *Beta,double *u, int Np,double *cMajor,double *rMajor,double *output,int Nt,int Kdim){
	//tested correct
	double *KtU = (double*) calloc(Nt,sizeof(double));
	//matrix*vector to calculate right hand side, once only
	
	// printf("U before comp:\n");
	// printMat(u,1,Np);
	// printf("output:\n");
	// printMat(output,1,Np);
	
	getOTminusKtU(KtU, u,Nt, Np, rMajor, output, Kdim );
	
	// printf("KtU after output-:\n");
	// printMat(KtU,1,Nt);
	
	// KtU is now the right hand side. 
	// maybe throw in a dgemv since it's vector matrix?
	for(int i=0;i<Np;i++){
		double temp = 0;
		for(int j=0;j<Nt;j++){
			temp += -1*cMajor[j+i*Kdim]*KtU[j];
		}
		Beta[i] = temp;
	}
	free(KtU);
}
void getAlpha(double *Alpha,double *u, int Np,double *cMajor,double *rMajor,double *output,int Nt,int Kdim){
	//tested correct...
	double *KtU = (double*) calloc(Nt,sizeof(double));
	//matrix*vector to calculate right hand side, once only
	// dgemv('N',Nt,Np,1.0,rMajor,Nt,u,1,1.0,KtU,1);
	
	getOTminusKtU(KtU, u,Nt, Np, rMajor,output, Kdim );
	
	// printf("KtU after output-:\n");
	// printMat(KtU,1,Nt);
	
	double temp=0;
	for(int i=0;i<Nt;i++){
		// KtU[i] = output[i]-KtU[i];
		temp += pow(KtU[i],2);
		// printf("KtUL%f, sq()=%f, temp: %f\n",KtU[i],pow(KtU[i],2),temp);
	}
	*Alpha = sqrt(temp);
	free(KtU);
}
//calculates the phi function
double phiOfU(double *U,int Np,double *cMajor,double *rMajor,double *output,int Nt,int Kdim){
	double uNorm=0;
	for(int i=0;i<Np;i++){
		uNorm += abs(U[i]);
	}
	double rightHandSide;
	getAlpha(&rightHandSide,U,Np,cMajor,rMajor,output,Nt,Kdim);
	return uNorm + rightHandSide;
}

double smallDeltaIZofU(double *cMajor, int i, int z, int Nt,int Kdim){
	//returns d_(i,z) of (u), but we don't need u
	double sum = 0;
	double *columnI = &(cMajor[i*Kdim]);
	double *columnZ = &(cMajor[z*Kdim]);
	//do manually rather than BLAS since we only want top Nt values, despite having dim values per column
	for(int j=0; j<Nt; j++){
		sum += columnI[j]*columnZ[j];
	}
	return sum;
}
double getHessianIZofU(double *cMajor, int i, int z, int Nt,int Kdim,double *Beta, double Alpha){
	//pass in Betas/alpha arrays already computed for given U
	double toret = smallDeltaIZofU(cMajor, i, z, Nt,Kdim) / Alpha;
	toret = toret - (Beta[i] * Beta[z])/pow(Alpha,3);
	return toret;
}
double getTrace(double *cMajor, int Nt,int Np, int Kdim, double *Beta, double Alpha){
	//pass in Betas/alpha arrays already computed for given U
	double sum=0;
	for (int i=0;i<Np;i++){
		sum += getHessianIZofU(cMajor, i, i, Nt, Kdim, Beta, Alpha);
	}
	return sum;
}
double getDeltaStep(double *cMajor, int Nt,int Np, int Kdim, double *Beta, double Alpha){
	//pass in Betas/alpha arrays already computed for given U
	return 1/sqrt( getTrace(cMajor, Nt, Np, Kdim, Beta, Alpha) );
}

void calcGrad(double *gradient, double *u, int Np,double *cMajor,double *rMajor,double *output,int Nt,int dim, double *Beta, double Alpha){
	//Gradient of the Phi objective function at the point u
	for (int i=0;i<Np;i++){
		if (u[i] > 0)
			gradient[i] = 1 + Beta[i]/Alpha;
		else
			gradient[i] = -1 + Beta[i]/Alpha;
	}
}
double getNorm(double *vect,int dim){
	//computes norm_1
	double sum = 0;
	for(int i=0;i<dim;i++){
		sum += abs(vect[i]);
	}
	return sum;
}

void updateUWithGrad(double *uIn, double *uOut, int Np, double *Gradient, int mSteps, double Delta_u){
	//uIn is preserved, uOut is overwritten with new U
	// calculates u = u-(Grad(u)*mSteps*Delta_u);
	for(int i=0;i<Np;i++){
		uOut[i] = uIn[i] - Gradient[i] * mSteps * Delta_u;
	}
}
double Obj(double *originalU,double *uToGetWrittenTo, int mSteps,int Np, double *Gradient, double Delta_u,double *cMajor,double *rMajor,double *output, int Nt,int Kdim){
	// returns { Phi( u-(Grad(u)*m*Delta_u) ) }
	// memcpy(newUSpace,u,Np*sizeof(double));
	updateUWithGrad(originalU,uToGetWrittenTo,Np,Gradient,mSteps,Delta_u);
	return phiOfU(uToGetWrittenTo,Np,cMajor,rMajor,output,Nt,Kdim);
}
//run l1 optimization
void calcL1(double *u,int Np,double *cMajor,double *rMajor,double *output,int Nt,int Kdim){
	double inTime = read_timer( );
	//initial U vector gets passed in....
	//workspace
	// create newTempUSpace so we don't destroy the previous U we've been passing around
	double *newTempUSpace = (double*)calloc(Np,sizeof(double));
	
	//calculate Alpha/Beta for this u
	double Alpha;
	double *Beta = (double*) calloc(Np,sizeof(double));
	getAlpha(&Alpha, u, Np,cMajor,rMajor,output,Nt,Kdim);
	getBeta(Beta,u, Np, cMajor,rMajor,output,Nt,Kdim);
	
		
	double *Gradient = (double*) calloc(Np,sizeof(double));
	calcGrad(Gradient, u, Np, cMajor, rMajor, output, Nt, Kdim, Beta, Alpha); //store gradient(u) in Gradient
	double Delta_u=999999999, m, mPlusOne;
	int loops = 0;
	// double norm = 999999999;//getNorm( Gradient, Np );
	double previousDelta_u; //high value to skip first Delta_u iteration
	double divisor = 100;
	// while( norm > EPSILON+100 ){ 
	while( true ){ 
		// if (loops++ % 10000){
			printf("----------------------------------------------\n");
			printf("Loops: %d \t time: %f\n",loops++,read_timer( )-inTime);
		// }
		printf("u:\n");
		// printMat(u,1,Np);
		printf("Grad:\n");
		// printMat(Gradient,1,Np);
		printf("Gradient norm: %f\n",getNorm( Gradient, Np ));
		printf("phi: %f\n",phiOfU(u,Np, cMajor, rMajor, output, Nt, Kdim));
		printf("Alpha: %f\n",Alpha);
		printf("Beta:\n");
		// printMat(Beta,1,Np);
		
		int mSteps = 0; 
		
		previousDelta_u = Delta_u;
		Delta_u = getDeltaStep(cMajor, Nt,Np, Kdim, Beta, Alpha)/divisor;
		while (Delta_u >= previousDelta_u){
			Delta_u = Delta_u/2;
		}
		
		printf("Delta_u: %f\n",Delta_u);
		// m = Obj(u,mSteps,Delta_u);
		m = Obj(u,newTempUSpace, mSteps,Np, Gradient, Delta_u,cMajor,rMajor,output, Nt,Kdim);
		// mPlusOne = Obj(u,mSteps+1,Delta_u);
		mPlusOne = Obj(u,newTempUSpace, mSteps+1,Np, Gradient, Delta_u,cMajor,rMajor,output, Nt,Kdim);
		double diff = abs(m - mPlusOne);
		printf("Step difference: %f\n",diff);
		if (diff <= EPSILON)
			break;
		
		printf("\t\tSteps:%d\tm=%f m+1=%f\n",mSteps,m,mPlusOne);
		while( true ){
			if ( m > mPlusOne ){
				mSteps++;
			}else{
				break;
			}
			m = mPlusOne; //save previous computation to prev holder, reduce computation by half
			// mPlusOne = Obj(u,mSteps,Delta_u); //is plus one, due to the mSteps++
			mPlusOne = Obj(u,newTempUSpace, mSteps+1,Np, Gradient, Delta_u,cMajor,rMajor,output, Nt,Kdim);
			//now mPlusOne is one mSteps ahead of m
			printf("\t\tSteps:%d\tm=%f m+1=%f\n",mSteps,m,mPlusOne);
		}
		
		printf("\t\tBroke after with mSteps: %d\n",mSteps);
		// u = u-(Grad(u)*mSteps*Delta_u);
		//Set uIn=uOut=u, since we want to overwrite the old u
		updateUWithGrad(u,u, Np, Gradient, mSteps, Delta_u);
		
		//recompute Alpha/Beta for this new u		
		getAlpha(&Alpha, u, Np,cMajor,rMajor,output,Nt,Kdim);
		getBeta(Beta,u, Np, cMajor,rMajor,output,Nt,Kdim);
		
		// Calculate Gradient for next timestep with next u
		calcGrad(Gradient, u, Np, cMajor, rMajor, output, Nt, Kdim, Beta, Alpha); //store gradient(u) in Gradient
		
		// norm = getNorm( Gradient, Np );
		
		//if (loops > 100)
		//	break;
	} 
	//now u == u*
	// u holds the optimal u
	free(Beta);
	free(Gradient);
	free(newTempUSpace);
	return;
}
//get parameter index for a given kernel
int paramInd(unsigned int *order,int ind){
	int n=0;
	for(int i=0;i<ind;i++){
		if (order[ind] == order[i])
			n++;
	}
	return n;
}

int main( int argc, char **argv ){ 

	// for timing purposes
    double simulation_time = read_timer( );
	
	if( find_option( argc, argv, "-h" ) >= 0 ){
		printUsage();
        return 0;
    }
	//get input/output file names
	char *inputFile = read_string( argc, argv, "-i", NULL );
	char *outputFile = read_string( argc, argv, "-o", NULL );
	char *kernelFile = read_string( argc, argv, "-f", NULL );
	
	if (inputFile == NULL || outputFile == NULL || kernelFile==NULL){
		printUsage();
        exit(1);
	}
	int maxNumberOfPointsToUse = read_int( argc, argv, "-m", -1 );
	if (maxNumberOfPointsToUse == -1){
		printUsage();
        exit(1);
	}
	
	//Set up parallel cores using MPI
	int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	
	int numInputPoints;
	int dimension=0;
	
	int numGaussian=0;
	int numLinear=0;
	int numPoly=0;
	
	// if (rank==0){
		// printf("Beginning time:\t%f\n",read_timer( )-simulation_time);
	// }
	int actuallyAre=0;
	if (rank==0){
		//only host core does this, get file info (#dimensions,#input points)
		printf("In: %s, out %s\n",inputFile,outputFile);
		unsigned int tempSize = 10000;
		char *temp = (char*)calloc(tempSize , sizeof(char));
		//figure out the # of input points
		FILE *fi = fopen(inputFile,"r");
		if (fi == NULL){
			printf("input file does not exist.\n");
			exit(-1);
		}
		fgets(temp,tempSize,fi);
		dimension++;
		actuallyAre = getNumIn(temp,",");
		while(fgets(temp,tempSize,fi)){
			dimension++;
		}
		fclose(fi);
		
		fi = fopen(kernelFile,"r");
		if (fi == NULL){
			printf("kernel file does not exist.\n");
			exit(-1);
		}
		//Get info on kernels...
		while(fgets(temp,tempSize,fi)){
			parseKernelInit( temp, ",",&numGaussian,&numLinear,&numPoly);
		}
		// printf("RANK0: G:%d, L:%d, P:%d\n",numGaussian,numLinear,numPoly);
		fclose(fi);
		free(temp);
		
		//do logic on how many input points to select
		numInputPoints = min(actuallyAre,maxNumberOfPointsToUse);
		printf("Selecting %d of %d points.\n",numInputPoints,actuallyAre);
	}
	
	//Send dimension/numInputPoints to all cores
	MPI_Bcast(&numInputPoints,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&dimension,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&numGaussian,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&numLinear,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&numPoly,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	// printf("R:%d G:%d, L:%d, P:%d\n",rank, numGaussian,numLinear,numPoly);
	// printf("R:%d d\n",rank);
	// fflush(stdout);
	double *gaussianParam = (double*)calloc(numGaussian,sizeof(double));
	double *polyC1 = (double*)calloc(numPoly,sizeof(double));
	double *polyC2 = (double*)calloc(numPoly,sizeof(double));
	double *polyD = (double*)calloc(numPoly,sizeof(double));
	
	
	//get max of dimension,#input points. Will be dimension of kernels...
	unsigned int maxDim = max(numInputPoints,dimension);
	unsigned int maxDimTotalLength = maxDim*maxDim;
	unsigned int rowsInTraining = ceil(maxDim/2);
	unsigned int rowsInTest = maxDim-rowsInTraining;
	//Check that all cores got them. Correct.
	// printf("R:%d numInputPoints:%d, dimension:%d\n",rank,numInputPoints,dimension);
	
	//Make space for the input data
	double *input = (double*) calloc(dimension*numInputPoints,sizeof(double));
	if (input == NULL){
		//make sure space was created
		printf("R:%d Error calloc'ing on input %d bytes\n",rank,dimension*numInputPoints);
		exit(1);
	}
	//make space for output data
	double *output = (double*) calloc(numInputPoints,sizeof(double));
	if (output == NULL){
		printf("R:%d Error calloc'ing on output\n",rank);
		exit(1);
	}
	
	// printf("R:%d e\n",rank);
	// fflush(stdout);
	if(rank == 0){
		//host cpu grabs data from files.
		
		//parse csv files
		unsigned int tempSize = 10000;
		char *temp = (char*)calloc(tempSize , sizeof(char));
		
	// printf("R:%d f\n",rank);
	// fflush(stdout);
		//Grab input csv
		FILE *fi = fopen(inputFile,"r");
		unsigned int c=0;
		while(fgets(temp,tempSize,fi)){
			parse( temp, ",",&(input[c*numInputPoints]),actuallyAre,numInputPoints);
			c++;
		}
		fclose(fi);
		// printf("input:\n");
		// printMat(input,numInputPoints,dimension);
		// printf("\n");
		
		//grab output csv
		FILE *fo = fopen(outputFile,"r");
		if (fo == NULL){
			printf("output file does not exist.\n");
			exit(-1);
		}
		fgets(temp,tempSize,fo);
		parse(temp, ",",output,actuallyAre,numInputPoints);
		fclose(fo);
		// printf("output:\n");
		// printMat(output,1,numInputPoints);
		// printf("\n");
		
		fi = fopen(kernelFile,"r");
		//Get info on kernels...
		
		// reset for next step
		numGaussian=0;
		numLinear=0;
		numPoly=0;
		
		while(fgets(temp,tempSize,fi)){
			parseKernel( temp, ",",&numGaussian,&numLinear,&numPoly,gaussianParam,polyC1,polyC2,polyD);
		}
		
		// printf("R:%d G:%d(%f) L:%d P:%d(%f,%f,%f)\n",rank,numGaussian,gaussianParam[0],numLinear,
		// numPoly,polyC1[0],polyC2[0],polyD[0]);
		fclose(fi);
		
		free(temp);
	}
	//broadcast input data to all cores : correct
	// printf("R:%d waiting/sending G:%d, P:%d\n",rank, numGaussian,numPoly);
	MPI_Bcast(input,dimension*numInputPoints,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(gaussianParam,numGaussian,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(polyC1,numPoly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(polyC2,numPoly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(polyD,numPoly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	MPI_Bcast(output,numInputPoints,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	unsigned int numKernels = numGaussian+numLinear+numPoly;
		
	// MPI_Barrier(MPI_COMM_WORLD);
	// return 0;
	
	if (rank==0){
		printf("Set up time:\t%f\n",read_timer( )-simulation_time);
	}
	
	//Create the same ordering of kernels on all procs (so all procs know who's doing what)
	unsigned int kernAss[numKernels];
	for(unsigned int i=0;i<numGaussian;i++){
		kernAss[i] = 0;
	}
	for(unsigned int i=0;i<numLinear;i++){
		kernAss[i+numGaussian] = 1;
	}
	for(unsigned int i=0;i<numPoly;i++){
		kernAss[i+numGaussian+numLinear] = 2;
	}
	// if (rank==0){
		// printf("a\n");
		// fflush(stdout);
	// }
	//get kernels for this rank to compute...
	//shuffle the job order, same seed on all procs. Load balances the kernels
	srand(n_proc);
	unsigned int t,j;
	for(unsigned int i=0;i<numKernels-1;i++){
		j = rand()%numKernels;
		t = kernAss[j];
		kernAss[j] = kernAss[i];
		kernAss[i] = t;
	}
	// if (rank==0){
		// printf("b\n");
		// fflush(stdout);
	// }
	//allocate space for most this proc would have to work on...
	int procWorkSize[n_proc];
	int recvOffset[n_proc];
	for(int i=0;i<n_proc;i++){
		procWorkSize[i] = (int)ceil(1.0*(numKernels-i)/n_proc)*maxDimTotalLength;
		if (i>0)
			recvOffset[i] = recvOffset[i-1]+procWorkSize[i-1];
		else
			recvOffset[i] = 0;
	}
	// if (rank==0){
		// printf("c\n");
		// fflush(stdout);
	// }
	double *workSpace = (double*)calloc(procWorkSize[rank],sizeof(double));
	int workedSoFar = 0;
	int paramIndex;
	// each proc does own kernels
	for(unsigned int i=rank;i<numKernels;i+=n_proc){
		paramIndex = paramInd(kernAss,i);
		if(kernAss[i] == 0){
			// printf("R:%d calculating gaussian %d\n",rank,i);
			calcGaussKernel(input,workSpace+(workedSoFar*maxDimTotalLength),dimension,numInputPoints,gaussianParam[paramIndex]);
		}else if(kernAss[i] == 1){
			// printf("R:%d calculating linear %d\n",rank,i);
			calcLinKernel(input,workSpace+(workedSoFar*maxDimTotalLength),dimension,numInputPoints);
		}else if(kernAss[i] == 2){
			// printf("R:%d calculating poly %d\n",rank,i);
			calcPolyKernel(input, workSpace+(workedSoFar*maxDimTotalLength),dimension,numInputPoints, polyC1[paramIndex],polyC2[paramIndex], polyD[paramIndex]);
		}
		workedSoFar++;
	}
	// if (rank==0){
		// printf("d\n");
		// fflush(stdout);
	// }
// printf("R:%d workLen:%d proc todo:%d done:%d sending #s:%d at offset:%d\n",
// rank,procWorkSize[rank],procWorkSize[rank]/maxDimTotalLength,workedSoFar,workedSoFar*maxDimTotalLength,recvOffset[rank]);
	
	double *fatKColumnMajor,*fatKRowMajor;
	//every proc get the fatK info
	fatKColumnMajor = (double*)calloc(maxDimTotalLength*numKernels,sizeof(double));
	fatKRowMajor = (double*)calloc(maxDimTotalLength*numKernels,sizeof(double));
	
	
	if (rank==0){
		printf("Calculation time:\t%f\n",read_timer( )-simulation_time);
	}
	
	//gather the kernels back to the main.
	// MPI_Gatherv ( workSpace,workedSoFar*maxDimTotalLength , MPI_DOUBLE, fatKColumnMajor, procWorkSize, recvOffset, MPI_DOUBLE, 0, MPI_COMM_WORLD );
	MPI_Allgatherv ( workSpace,workedSoFar*maxDimTotalLength , MPI_DOUBLE, fatKColumnMajor, procWorkSize, recvOffset, MPI_DOUBLE, MPI_COMM_WORLD );
		
	
	//Fat k is now in column major. 
	// easiest to bring together in column, then transpose. easier than interleaving portions...
	if (rank==0){
		printf("Gather time:\t%f\n",read_timer( )-simulation_time);
	}
	
	free(workSpace);
	
	//Throw fat k into row major to establish training/test vectors
	// Blocked tranpose
	int blockSize = min(BLOCKSIZE,(int)maxDim);
	for(int y=0;y<((int)ceil(1.0*maxDim/blockSize));y++){
		for(int x=0;x<ceil(1.0*maxDim*numKernels/blockSize);x++){
			for(int i=y*blockSize;i<y*blockSize+blockSize&&i<maxDim;i++){
				for(int j=x*blockSize;j<x*blockSize+blockSize&&j<maxDim*numKernels;j++){
					fatKRowMajor[j+i*maxDim*numKernels] = fatKColumnMajor[i+j*maxDim];
				}
			}
		}
	}		
		
	if (rank==0){
		// printf("fatKRowMajor:\n");
		// printMat(fatKRowMajor,maxDim,maxDim*numKernels);
		// printf("fatKColumnMajor:\n");
		// printMat(fatKColumnMajor,maxDim,maxDim*numKernels);
		// printf("reformat fatKColumnMajor:\t%f\n",read_timer( )-simulation_time);
	}
	
	
	int Np = numKernels*maxDim;
	// have each slave proc calculate and send back result
	double *U = (double*) calloc (Np,sizeof(double));
	if (rank != 0){
		//Calculate own starting location...
		srand(rank); //unique seed for each proc, same seed=same starting locations...
		for(int i=0;i<Np;i++){
			//Begin with random starting location
			U[i] = (1.0*rand()+1)/RAND_MAX; //0.1;
		}
		//optimize L1
		
		calcL1(U,Np,fatKColumnMajor,fatKRowMajor,output,rowsInTraining,maxDim);
		printf("R: %d\tOptimal U on slave:\n",rank);
		printMat(U,1,Np);
		printf("\n");
		fflush(stdout);
		
		// now U holds the optimal u
		MPI_Send( U, Np, MPI_DOUBLE, 0, L1RESULT , MPI_COMM_WORLD );
	}else{
		
		//working on implementing an efficient "get the first response" algorithm
		
		//holds numProcs-1 sets of Us
		double *holder = (double*) calloc ((n_proc-1)*(Np),sizeof(double));
		MPI_Request *requests = (MPI_Request*) calloc (n_proc-1,sizeof(MPI_Request));
		for(int i=0;i<(n_proc-1);i++){
			// set up nonblocking recvs...
			MPI_Irecv(holder+i*Np, Np, MPI_DOUBLE, i+1, L1RESULT, MPI_COMM_WORLD, &(requests[i]) );
		}
		int c = 0;
		int flag = 0;
		// printf("checking c:%d\n",c);
		MPI_Request_get_status(requests[c], &flag, MPI_STATUS_IGNORE);
		unsigned int iter=0;
		while (flag == 0){
			iter++;
			c = c++ % (n_proc-1);
			// printf("checking c:%d\n",c);
			MPI_Request_get_status(requests[c], &flag, MPI_STATUS_IGNORE);
		}
		//c now is index to the first received U
		// printf("First index: %d after %u iters\n",c+1,iter);
		
		//Assume U now holds the optimal answer from some proc
		memcpy(U, &(holder[c]),Np*sizeof(double));
		/*
		double phiStar = phiOfU(U,numKernels*maxDim,fatKColumnMajor,fatKRowMajor,output,rowsInTraining,maxDim);
		//get vStar
		double vStar;
		getAlpha(&vStar,U, numKernels*maxDim,fatKColumnMajor,fatKRowMajor,output,rowsInTraining,maxDim);
		//get muStar
		double muStar = vStar/phiStar;
		
		
		
		//build KStar
		double *KStar = (double*)calloc(maxDim*maxDim,sizeof(double));
		double *currentColumn;
		for(int j=0;j<numKernels*maxDim;j++){
			//iterate through each column of K, each eigenvector
			currentColumn = &(fatKColumnMajor[j]);
			double thisLambda = abs(U[j])/phiStar;
			//multiply column* Trans(column) to get square matrix
			//since we're summing matrices, can simply sum each element at a given location for the matrix?
			for(int x=0;x<maxDim;x++){
				for(int y=0;y<maxDim;y++){
					KStar[y+x*maxDim] += currentColumn[x]*currentColumn[y]*thisLambda;
				}
			}
		}
		
		// Now KStar is built
		
		double *KTrainingStar = (double*) calloc(rowsInTraining*rowsInTraining, sizeof(double));
		//copy gets trashed by dgetrf()
		double *KTrainingStarCopy = (double*) calloc(rowsInTraining*rowsInTraining, sizeof(double));
		for(int i=0;i<rowsInTraining;i++){
			for(int j=0;j<rowsInTraining;j++){
				if (i==j){
					//diagonal
					KTrainingStar[j+i*rowsInTraining] += muStar;
				}
				KTrainingStar[j+i*rowsInTraining] += KStar[j+i*maxDim]; //limit by maxDim since KStar is NxN
				KTrainingStarCopy[j+i*rowsInTraining] = KTrainingStar[j+i*rowsInTraining];
			}
		}
		// KTrainingStar is built
		int rowsInTest = maxDim - rowsInTraining;
		double *XStar = (double*) calloc(rowsInTest*rowsInTraining, sizeof(double));
		for(int i=0;i<rowsInTraining;i++){
			for(int j=0;j<rowsInTest;j++){
				//limit by maxDim since KStar is NxN
				//add rowsInTraining to skip Nt section
				XStar[j+i*rowsInTest] += KStar[j+rowsInTraining+i*maxDim]; 
			}
		}
		//XStar is built
		
		int *pivotInfo = (int*)calloc(rowsInTraining,sizeof(int));
		int success=0;
		dgetrf( rowsInTraining, rowsInTraining, KTrainingStarCopy, rowsInTraining, pivotInfo, &success );
		if (success != 0){
			printf("DGETRF failed?\n");
			exit(-1);
		}
		double *KTrainingStarInverse = KTrainingStar; //for naming ease
		//Do the inverse computation, store in KTrainingStar
		dgetri( rowsInTraining,KTrainingStarInverse,rowsInTraining,pivotInfo,&success);
		if (success != 0){
			printf("DGETRI failed?\n");
			exit(-1);
		}
		
		double *alphaStars = (double*) calloc(rowsInTraining,sizeof(double));
		//compute KTrainingStarInverse * OutputTrainingTranspose
		dgemv('N',rowsInTraining,rowsInTraining,1.0,KTrainingStarInverse,rowsInTraining,
					output,1,1.0,alphaStars,1);
		
		double *estimates = (double*)calloc(rowsInTest,sizeof(double));
		dgemv('N',rowsInTest,rowsInTraining,1.0,XStar,rowsInTest,
					estimates,1,1.0,estimates,1);
		
		printf("Estimates: \n");
		printMat(estimates,rowsInTest,1);
		printf("\n\n");
		
		free(KStar);
		free(KTrainingStar);
		free(KTrainingStarCopy);
		free(XStar);
		free(pivotInfo);
		free(alphaStars);
		free(estimates);
		*/
	}
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);
	// if (rank == 0){
		// printf("\n");
		// printf("\n");
		// printf("Optimal U on host:\n");
		// printMat(U,1,Np);
		// printf("\n");
		// printf("\n");
	// }
	MPI_Barrier(MPI_COMM_WORLD);
    if( rank == 0 ){
        printf("input:%dx%d n_procs = %d, simulation time = %g s\n", dimension,numInputPoints, n_proc, read_timer( )-simulation_time );
		printf("NumKernels: %d\n",numKernels);	
		printf("Linear: used:%d\n",numLinear);
		printf("Polynomial: used:%d\tc1:%f,c2:%f,d:%f\n",numPoly,polyC1[0],polyC2[0],polyD[0]);
		printf("Gaussian: used:%d\tsigma:%f\n",numGaussian,gaussianParam[0]);
		printf("fatKColumnMajor: %u values, %u bytes\n",maxDimTotalLength*numKernels,maxDimTotalLength*numKernels*sizeof(double));
		printf("Training: %d rows, Testing: %d rows\n",rowsInTraining,rowsInTest);
    }
	
	//Free all calloc'ed space
	free(U);
	free(fatKColumnMajor);
	free(fatKRowMajor);
	free(input);
	free(output);
	//Necessary for actual completion...
	MPI_Barrier(MPI_COMM_WORLD);
	
	// printf("R:%d Finishing application\n",rank);
	// fflush(stdout);
	return 0;
}
