#include <time.h>
#define TIMING 1


#if TIMING == 1
#define TIMEINIT float milliseconds, total;\
cudaEvent_t start, stop;\
cudaEventCreate(&start);\
cudaEventCreate(&stop);\
milliseconds = total = 0;

#define TIMEIT(f, name) cudaEventRecord(start);\
f;\
cudaEventRecord(stop);\
cudaEventSynchronize(stop);\
cudaEventElapsedTime(&milliseconds, start, stop);\
printf("%s Elapsed Time: %f\n", name, milliseconds);\
total += milliseconds;

#define TIMEEND printf("Total Elapsed Time: %f\n", total);

#define CPUTIMEINIT clock_t start, end;
#define CPUTIMEIT(f, name) start = clock();\
f;\
end = clock() - start;\
printf("%s Elapsed Time: %f\n", name, float(end)/1000.0f);


#else

#define TIMEINIT
#define TIMEIT(f, name) f;
#define TIMEEND
#define CPUTIMEINIT
#define CPUTIMEIT(f, name) f;
#endif
