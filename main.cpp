#include <iostream>
#include <fstream>

#include <neural.h>

std::ifstream fin;

//Number of layers and size of each layer (bias excluded)
int L, *layers;

//Number of training inputs and outputs
int N;
double **inputs, **outputs;

int main(int argc, char **argv){
	if(argc!=3){
		std::cout<<"[USAGE] neural {training files}\n";
		return 0;
	}
	
	fin.open(argv[1]);
	
	//Read network structure
	fin>>L;
	layers=new int[L];
	for(int i=0;i<L;i++)
		fin>>layers[i];
	
	//Initialize arrays and random weights
	init_network();
	
	//Read training sets
	fin>>N;
	inputs=new double*[N];
	outputs=new double*[N];
	for(int i=0;i<N;i++){
		inputs[i]=new double[layers[0]];
		outputs[i]=new double[layers[L-1]];
		for(int j=0;j<layers[0];j++)
			fin>>inputs[i][j];
		for(int j=0;j<layers[L-1];j++)
			fin>>outputs[i][j];
	}
	
	std::cout<<"TEST DATA\n";
	for(int i=0;i<N;i++){
		std::cout<<"["<<(i+1)<<"]:\tIN : ";
		for(int j=0;j<layers[0];j++)
			std::cout<<inputs[i][j]<<" ";
		std::cout<<"\n";
		std::cout<<"\tOUT: ";
		for(int j=0;j<layers[L-1];j++)
			std::cout<<outputs[i][j]<<" ";
		std::cout<<"\n";
	}
	
	//Train until all the testcases are learned
	std::cout<<"TRAINING\n";
	train(inputs, outputs, 3, 0.00004, N, 1);
	
	double *out=new double[layers[L-1]];
	for(int i=0;i<N;i++){
		out=feed_forward(inputs[i]);
		std::cout<<"["<<(i+1)<<"]:\tEXPECTED: ";
		for(int j=0;j<layers[L-1];j++)
			std::cout<<outputs[i][j]<<" ";
		std::cout<<"\n\tOUTPUT: ";
		for(int j=0;j<layers[L-1];j++)
			std::cout<<out[j]<<" ";
		std::cout<<"\n";
	}
	
	fin.close();
}
