#include <neural.h>
#include <sigmoid.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

double ***weights, **out, **bias, **delta;

void init_network(){
	//Initialize neuron output array
	out=new double*[L];
	for(int i=0;i<L;i++)
		out[i]=new double[layers[i]];
	
	//Randomly initialize weights
	srand(time(NULL));
	weights=new double**[L];
	for(int i=1;i<L;i++){
		weights[i]=new double*[layers[i]];
		for(int j=0;j<layers[i];j++){
			weights[i][j]=new double[layers[i-1]];
			for(int k=0;k<layers[i-1];k++)
				weights[i][j][k]=(double)(rand()%1000-500)/1000;
		}
	}
	
	//Randomly initialize biases
	bias=new double*[L];
	for(int i=1;i<L;i++){
		bias[i]=new double[layers[i]];
		for(int j=0;j<layers[i];j++)
			bias[i][j]=(double)(rand()%1000-500)/1000;
	}
	
	//Initialize delta array
	delta=new double*[L];
	for(int i=0;i<L;i++){
		delta[i]=new double[layers[i]];
		for(int j=0;j<layers[i];j++)
			delta[i][j]=0;
	}
}

double* feed_forward(double *input){
	//Enter input into the network
	for(int i=0;i<layers[0];i++)
		out[0][i]=input[i];
	//Propagate through layers
	for(int i=1;i<L;i++){
		//For each neuron
		for(int j=0;j<layers[i];j++){
			//Weighted sum of last layer's outputs
			double w_sum=0;
			for(int k=0;k<layers[i-1];k++)
				w_sum+=sigma(out[i-1][k])*weights[i][j][k];
			out[i][j]=w_sum+bias[i][j];
		}
	}
	//Apply sigma on outputs since the array stores the unprocessed values
	double *output=new double[layers[L-1]];
	for(int i=0;i<layers[L-1];i++)
		output[i]=sigma(out[L-1][i]);
	return output;
}

double calculate_error(double *expected, double *out){
	double C=0;
	for(int i=0;i<layers[L-1];i++)
		C+=(expected[i]-out[i])*(expected[i]-out[i])/2;
	return C;
}

void flush_deltas(){
	for(int i=0;i<L;i++)
		for(int j=0;j<layers[i];j++)
			delta[i][j]=0;
}

void back_propagate(double *exp){
	double **new_delta=new double*[L];
	for(int i=0;i<L;i++)
		new_delta[i]=new double[layers[i]];
	//Final layer
	for(int i=0;i<layers[L-1];i++)
		new_delta[L-1][i]=(sigma(out[L-1][i])-exp[i])*sigma_prime(out[L-1][i]);
	//Backpropagate
	for(int i=L-2;i>=0;i--){
		for(int j=0;j<layers[i];j++){
			double w_sum=0;
			for(int k=0;k<layers[i+1];k++)
				w_sum+=weights[i+1][k][j]*new_delta[i+1][k];
			new_delta[i][j]=w_sum*sigma_prime(out[i][j]);
		}
	}
	//Update delta
	for(int i=0;i<L;i++)
		for(int j=0;j<layers[i];j++)
			delta[i][j]+=new_delta[i][j];
}

void update_weights(double rate){
	for(int i=1;i<L;i++){
		//Update layer weights
		for(int j=0;j<layers[i];j++)
			for(int k=0;k<layers[i-1];k++)
				weights[i][j][k]-=rate*sigma(out[i-1][k])*delta[i][j];
		//Update biases
		for(int j=0;j<layers[i];j++)
			bias[i][j]-=rate*delta[i][j];
	}
}

/*
 * NOTE: the mini batch size must divide the total number of training examples
 */
void train(double **inputs, double **outputs, double rate,
		double target_error, int set_size, int mini_batch_size){
	//Repeat until results are good enough
	double err[set_size/mini_batch_size];
	double *o=new double[layers[L-1]];
	bool trained;
	int epoch=0;
	do{
		epoch++;
		for(int i=0;i<set_size/mini_batch_size;i++)
			err[i]=0;
		trained=true;
		//Take a mini-batch at a time
		for(int i=0;i<set_size;i+=mini_batch_size){
			//Backpropagate on the mini-batch's examples
			double e=0;
			for(int j=0;j<mini_batch_size;j++){
				o=feed_forward(inputs[i+j]);
				err[i/mini_batch_size]+=calculate_error(outputs[i+j], o)/mini_batch_size;
				back_propagate(outputs[i+j]);
			}
			std::cout<<err[i/mini_batch_size]<<" ";
			//Update weights and check error
			update_weights(rate);
			flush_deltas();
			if(err[i/mini_batch_size]>target_error)
				trained=false;
		}
		std::cout<<"\n";
	}
	while(!trained);
	
	std::cout<<"Took "<<epoch<<" epochs to get it right\n";
}
