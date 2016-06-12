#ifndef __SIGMOID_H
#define __SIGMOID_H

#include <cmath>

double sigma(double x){
	return 1/(1+exp(-x));
}

double sigma_prime(double x){
	return sigma(x)*(1-sigma(x));
}

#endif
