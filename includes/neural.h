#ifndef __NEURAL_H
#define __NEURAL_H

extern double ***weights, **out, **bias;
extern int L, *layers;

void init_network();
void flush_deltas();

double* feed_forward(double *input);

double calculate_error(double *expected, double *out);

void back_propagate(double *exp);
void update_weights(double rate);

void train(double **inputs, double **outputs, double rate,
		double target_error, int set_size, int mini_batch_size);

#endif
