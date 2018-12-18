/******************************************************************************
 *                       Code generated with sympy 1.3                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/


#ifndef PRECIPITATEAGING__NUCLEATION__H
#define PRECIPITATEAGING__NUCLEATION__H

double k1(double D, double F, double L, double T, double W, double a, double c0, double gamma, double k, double v);
double k2(double F, double T, double a, double gamma, double k);
double bell_curve(double sigma, double x);
double bell_integral(double sigma, double x);
double bell_average(double a, double b, double sigma);

#endif

