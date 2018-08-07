/******************************************************************************
 *                      Code generated with sympy 1.1.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/
#include "parabola625.h"
#include <math.h>

__device__ double d_h(double x) {

   double h_result;
   h_result = pow(x, 3)*(6.0*pow(x, 2) - 15.0*x + 10.0);
   return h_result;

}

__device__ double d_hprime(double x) {

   double hprime_result;
   hprime_result = 30.0*pow(x, 2)*pow(-x + 1.0, 2);
   return hprime_result;

}

__device__ double d_xe_gam_Cr() {

   double xe_gam_Cr_result;
   xe_gam_Cr_result = 0.49;
   return xe_gam_Cr_result;

}

__device__ double d_xe_gam_Nb() {

   double xe_gam_Nb_result;
   xe_gam_Nb_result = 0.025;
   return xe_gam_Nb_result;

}

__device__ double d_xe_del_Cr() {

   double xe_del_Cr_result;
   xe_del_Cr_result = 0.015;
   return xe_del_Cr_result;

}

__device__ double d_xe_del_Nb() {

   double xe_del_Nb_result;
   xe_del_Nb_result = 0.245;
   return xe_del_Nb_result;

}

__device__ double d_xe_lav_Cr() {

   double xe_lav_Cr_result;
   xe_lav_Cr_result = 0.3;
   return xe_lav_Cr_result;

}

__device__ double d_xe_lav_Nb() {

   double xe_lav_Nb_result;
   xe_lav_Nb_result = 0.328;
   return xe_lav_Nb_result;

}

__device__ double d_Vm() {

   double Vm_result;
   Vm_result = 1.0e-5;
   return Vm_result;

}

__device__ double d_xr_gam_Cr(double P_del, double P_lav) {

   double xr_gam_Cr_result;
   xr_gam_Cr_result = -1.35745662070142e-9*P_del + 6.39244600941687e-10*P_lav + 0.49;
   return xr_gam_Cr_result;

}

__device__ double d_xr_gam_Nb(double P_del, double P_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 2.47594876167551e-10*P_del - 6.66222300559304e-11*P_lav + 0.025;
   return xr_gam_Nb_result;

}

__device__ double d_xr_del_Cr(double P_del, double P_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = -4.44614260967884e-11*P_del + 2.79333077660415e-11*P_lav + 0.015;
   return xr_del_Cr_result;

}

__device__ double d_xr_del_Nb(double P_del, double P_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = -5.4311181652809e-12*P_del + 2.05131716667751e-11*P_lav + 0.245;
   return xr_del_Nb_result;

}

__device__ double d_xr_lav_Cr(double P_del, double P_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = -1.42576885211288e-10*P_del + 1.05186220574504e-10*P_lav + 0.3;
   return xr_lav_Cr_result;

}

__device__ double d_xr_lav_Nb(double P_del, double P_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = -7.60450759974964e-12*P_del + 1.63559527488494e-11*P_lav + 0.328;
   return xr_lav_Nb_result;

}

__device__ double d_inv_fict_det(double f_del, double f_gam, double f_lav) {

   double inv_fict_det_result;
   inv_fict_det_result = 1.0/(1.15317496889714*f_del*f_del + 42.9640255033086*f_del*f_gam + 4.35012289609114*f_del*f_lav + 102.885881033764*f_gam*f_gam + 51.3939292669239*f_gam*f_lav + 2.32367001744861*f_lav*f_lav);
   return inv_fict_det_result;

}

__device__ double d_fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = INV_DET*(36.9670932572906*XCR*f_del + 102.885881033764*XCR*f_gam + 26.3010348077371*XCR*f_lav - 14.4030540553513*XNB*f_del - 78.3270137345168*XNB*f_lav + 3.53929757946133*f_del*f_del + 1.75528493642612*f_del*f_gam + 14.5612368148933*f_del*f_lav - 16.6120706817648*f_gam*f_lav + 18.9395483711502*f_lav*f_lav);
   return fict_gam_Cr_result;

}

__device__ double d_fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = INV_DET*(-7.15429731355127*XCR*f_del - 5.37357718050628*XCR*f_lav + 5.99693224601796*XNB*f_del + 102.885881033764*XNB*f_gam + 25.0928944591868*XNB*f_lav - 1.33310456634871*f_del*f_del - 20.7772578381999*f_del*f_gam - 5.77910699501942*f_del*f_lav - 30.4559902904332*f_gam*f_lav - 6.56030447802516*f_lav*f_lav);
   return fict_gam_Nb_result;

}

__device__ double d_fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = INV_DET*(1.15317496889714*XCR*f_del + 5.99693224601796*XCR*f_gam + 0.780764088654332*XCR*f_lav + 14.4030540553513*XNB*f_gam - 1.05269525818007*XNB*f_lav - 3.53929757946133*f_del*f_gam - 0.0345017703034707*f_del*f_lav - 1.75528493642612*f_gam*f_gam - 6.10862948694289*f_gam*f_lav + 0.145909868348492*f_lav*f_lav);
   return fict_del_Cr_result;

}

__device__ double d_fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = INV_DET*(7.15429731355127*XCR*f_gam - 0.101861460320036*XCR*f_lav + 1.15317496889714*XNB*f_del + 36.9670932572906*XNB*f_gam + 3.56935880743681*XNB*f_lav + 1.33310456634871*f_del*f_gam - 0.185426266173148*f_del*f_lav + 20.7772578381999*f_gam*f_gam - 1.71930496668943*f_gam*f_lav - 0.57089209646835*f_lav*f_lav);
   return fict_del_Nb_result;

}

__device__ double d_fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = INV_DET*(3.56935880743681*XCR*f_del + 25.0928944591868*XCR*f_gam + 2.32367001744861*XCR*f_lav + 1.05269525818007*XNB*f_del + 78.3270137345168*XNB*f_gam + 0.0345017703034707*f_del*f_del - 8.45260732795046*f_del*f_gam - 0.145909868348492*f_del*f_lav + 16.6120706817648*f_gam*f_gam - 18.9395483711502*f_gam*f_lav);
   return fict_lav_Cr_result;

}

__device__ double d_fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = INV_DET*(0.101861460320036*XCR*f_del + 5.37357718050628*XCR*f_gam + 0.780764088654332*XNB*f_del + 26.3010348077371*XNB*f_gam + 2.32367001744861*XNB*f_lav + 0.185426266173148*f_del*f_del + 7.49841196170884*f_del*f_gam + 0.57089209646835*f_del*f_lav + 30.4559902904332*f_gam*f_gam + 6.56030447802516*f_gam*f_lav);
   return fict_lav_Nb_result;

}

__device__ double d_r_delta() {

   double r_delta_result;
   r_delta_result = 7.5e-9;
   return r_delta_result;

}

__device__ double d_r_laves() {

   double r_laves_result;
   r_laves_result = 7.5e-9;
   return r_laves_result;

}

__device__ double d_s_delta() {

   double s_delta_result;
   s_delta_result = 1.01;
   return s_delta_result;

}

__device__ double d_s_laves() {

   double s_laves_result;
   s_laves_result = 1.011;
   return s_laves_result;

}

__device__ double d_g_gam(double XCR, double XNB) {

   double g_gam_result;
   g_gam_result = 2474387391.41825*pow(XCR - 0.49, 2) + (15148919424.3273*XCR - 7422970517.92036)*(XNB - 0.025) + 37770442067.0002*pow(XNB - 0.025, 2);
   return g_gam_result;

}

__device__ double d_g_del(double XCR, double XNB) {

   double g_del_result;
   g_del_result = 32328981887.7551*pow(XCR - 0.015, 2) + (16970167763.7333*XCR - 254552516.456)*(XNB - 0.245) + 101815797663.265*pow(XNB - 0.245, 2);
   return g_del_result;

}

__device__ double d_g_lav(double XCR, double XNB) {

   double g_lav_result;
   g_lav_result = (-3224431198.33484*XCR + 967329359.500452)*(XNB - 0.328) + 10490734425.5087*pow(XCR - 0.3, 2) + 152553892719.717*pow(XNB - 0.328, 2);
   return g_lav_result;

}

__device__ double d_dg_gam_dxCr(double XCR, double XNB) {

   double dg_gam_dxCr_result;
   dg_gam_dxCr_result = 4948774782.83651*XCR + 15148919424.3273*XNB - 2803622629.19807;
   return dg_gam_dxCr_result;

}

__device__ double d_dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 15148919424.3273*XCR + 75540884134.0004*XNB - 9311492621.27037;
   return dg_gam_dxNb_result;

}

__device__ double d_dg_del_dxCr(double XCR, double XNB) {

   double dg_del_dxCr_result;
   dg_del_dxCr_result = 64657963775.5102*XCR + 16970167763.7333*XNB - 5127560558.74732;
   return dg_del_dxCr_result;

}

__device__ double d_dg_del_dxNb(double XCR, double XNB) {

   double dg_del_dxNb_result;
   dg_del_dxNb_result = 16970167763.7333*XCR + 203631595326.53*XNB - 50144293371.456;
   return dg_del_dxNb_result;

}

__device__ double d_dg_lav_dxCr(double XCR, double XNB) {

   double dg_lav_dxCr_result;
   dg_lav_dxCr_result = 20981468851.0174*XCR - 3224431198.33484*XNB - 5236827222.25138;
   return dg_lav_dxCr_result;

}

__device__ double d_dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = -3224431198.33484*XCR + 305107785439.433*XNB - 99108024264.6336;
   return dg_lav_dxNb_result;

}

__device__ double d_d2g_gam_dxCrCr() {

   double d2g_gam_dxCrCr_result;
   d2g_gam_dxCrCr_result = 4948774782.83651;
   return d2g_gam_dxCrCr_result;

}

__device__ double d_d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 15148919424.3273;
   return d2g_gam_dxCrNb_result;

}

__device__ double d_d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 15148919424.3273;
   return d2g_gam_dxNbCr_result;

}

__device__ double d_d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 75540884134.0004;
   return d2g_gam_dxNbNb_result;

}

__device__ double d_d2g_del_dxCrCr() {

   double d2g_del_dxCrCr_result;
   d2g_del_dxCrCr_result = 64657963775.5102;
   return d2g_del_dxCrCr_result;

}

__device__ double d_d2g_del_dxCrNb() {

   double d2g_del_dxCrNb_result;
   d2g_del_dxCrNb_result = 16970167763.7333;
   return d2g_del_dxCrNb_result;

}

__device__ double d_d2g_del_dxNbCr() {

   double d2g_del_dxNbCr_result;
   d2g_del_dxNbCr_result = 16970167763.7333;
   return d2g_del_dxNbCr_result;

}

__device__ double d_d2g_del_dxNbNb() {

   double d2g_del_dxNbNb_result;
   d2g_del_dxNbNb_result = 203631595326.53;
   return d2g_del_dxNbNb_result;

}

__device__ double d_d2g_lav_dxCrCr() {

   double d2g_lav_dxCrCr_result;
   d2g_lav_dxCrCr_result = 20981468851.0174;
   return d2g_lav_dxCrCr_result;

}

__device__ double d_d2g_lav_dxCrNb() {

   double d2g_lav_dxCrNb_result;
   d2g_lav_dxCrNb_result = -3224431198.33484;
   return d2g_lav_dxCrNb_result;

}

__device__ double d_d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = -3224431198.33484;
   return d2g_lav_dxNbCr_result;

}

__device__ double d_d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 305107785439.433;
   return d2g_lav_dxNbNb_result;

}
