#include "timer.h"
#include <time.h>


#define N 64
static struct timespec start_time[N];

void timer_start(int i) {
	clock_gettime(CLOCK_REALTIME, &start_time[i]);
}

double timer_stop(int i) {
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (tp.tv_sec - start_time[i].tv_sec) + (tp.tv_nsec - start_time[i].tv_nsec) * 1e-9;
}

struct timespec u_time;

double get_time() {
	clock_gettime(CLOCK_REALTIME, &u_time);
	return (u_time.tv_sec) + (u_time.tv_nsec) * 1e-9;
}	