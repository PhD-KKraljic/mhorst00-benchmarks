#include "wtime.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/*  Prototype  */
void wtime( double * );


/*****************************************************************/
/******         E  L  A  P  S  E  D  _  T  I  M  E          ******/
/*****************************************************************/
static double elapsed_time( void )
{
    double t;

#if defined(_OPENMP) && (_OPENMP > 200010)
/*  Use the OpenMP timer if we can */
    t = omp_get_wtime();
#else
    wtime( &t );
#endif
    return( t );
}


static double start[64], elapsed[64];
static unsigned count[64];
#ifdef _OPENMP
#pragma omp threadprivate(start, elapsed, count)
#endif

/*****************************************************************/
/******            T  I  M  E  R  _  C  L  E  A  R          ******/
/*****************************************************************/
void timer_clear( int n )
{
    elapsed[n] = 0.0;
    count[n] = 0;
}


/*****************************************************************/
/******            T  I  M  E  R  _  S  T  A  R  T          ******/
/*****************************************************************/
void mg_timer_start( int n )
{
    start[n] = elapsed_time();
}


/*****************************************************************/
/******            T  I  M  E  R  _  S  T  O  P             ******/
/*****************************************************************/
void mg_timer_stop( int n )
{
    double t, now;

    now = elapsed_time();
    t = now - start[n];
    elapsed[n] += t;
    count[n]++;

}


/*****************************************************************/
/******            T  I  M  E  R  _  R  E  A  D             ******/
/*****************************************************************/
double timer_read( int n )
{
    return( elapsed[n] );
}

unsigned timer_count( int n )
{
    return count[n];
}

