/******************************************************************************
 *                       Code generated with sympy 1.3                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/
#include "nucleation.h"
#include <math.h>

double k1(double D, double F, double L, double T, double W, double a, double c0, double gamma, double k, double v) {

   double k1_result;
   k1_result = (2.0/3.0)*sqrt(3)*D*L*W*c0*gamma*sqrt(pow(v, 2)*pow(F*a - 2*gamma, 3)/(T*pow(a, 4)*pow(gamma, 2)*k))*(F*a - gamma)/(a*v*pow(F*a - 2*gamma, 2));
   return k1_result;

}

double k2(double F, double T, double a, double gamma, double k) {

   double k2_result;
   k2_result = M_PI*pow(a, 2)*pow(gamma, 2)/(T*k*(F*a - 2*gamma));
   return k2_result;

}

double bell_curve(double sigma, double x) {

   double bell_curve_result;
   bell_curve_result = exp(-0.5*pow(x, 2)/pow(sigma, 2));
   return bell_curve_result;

}

double bell_integral(double sigma, double x) {

   double bell_integral_result;
   bell_integral_result = 0.70710678118654746*sqrt(M_PI)*sigma*erf(0.70710678118654757*x/sigma);
   return bell_integral_result;

}

double bell_average(double a, double b, double sigma) {

   double bell_average_result;
   bell_average_result = 0.70710678118654746*sqrt(M_PI)*sigma*(-erf(0.70710678118654757*a/sigma) + erf(0.70710678118654757*b/sigma))/(-a + b);
   return bell_average_result;

}
