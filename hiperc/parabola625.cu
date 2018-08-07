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
   h_result = x*x*x*(6.0*x*x - 15.0*x + 10.0);
   return h_result;

}

__device__ double d_hprime(double x) {

   double hprime_result;
   hprime_result = 30.0*x*x*(-x + 1.0)*(-x + 1.0);
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
   xr_gam_Cr_result = -1.35745662070135e-9*P_del + 6.39244600941618e-10*P_lav + 0.49;
   return xr_gam_Cr_result;

}

__device__ double d_xr_gam_Nb(double P_del, double P_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 2.4759487616754e-10*P_del - 6.6622230055919e-11*P_lav + 0.025;
   return xr_gam_Nb_result;

}

__device__ double d_xr_del_Cr(double P_del, double P_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = -4.44614260967858e-11*P_del + 2.79333077660391e-11*P_lav + 0.015;
   return xr_del_Cr_result;

}

__device__ double d_xr_del_Nb(double P_del, double P_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = -5.4311181652798e-12*P_del + 2.05131716667744e-11*P_lav + 0.245;
   return xr_del_Nb_result;

}

__device__ double d_xr_lav_Cr(double P_del, double P_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = -1.42576885211279e-10*P_del + 1.05186220574496e-10*P_lav + 0.3;
   return xr_lav_Cr_result;

}

__device__ double d_xr_lav_Nb(double P_del, double P_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = -7.60450759974872e-12*P_del + 1.63559527488487e-11*P_lav + 0.328;
   return xr_lav_Nb_result;

}

__device__ double d_inv_fict_det(double f_del, double f_gam, double f_lav) {

   double inv_fict_det_result;
   inv_fict_det_result = 1.0/(0.184507995023542*f_del*f_del + 6.87424408052938*f_del*f_gam + 0.696019663374582*f_del*f_lav + 16.4617409654023*f_gam*f_gam + 8.22302868270782*f_gam*f_lav + 0.371787202791778*f_lav*f_lav);
   return inv_fict_det_result;

}

__device__ double d_fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = 1.0*INV_DET*(5.9147349211665*XCR*f_del + 16.4617409654023*XCR*f_gam + 4.20816556923794*XCR*f_lav - 2.30448864885622*XNB*f_del - 12.5323221975227*XNB*f_lav + 0.566287612713812*f_del*f_del + 0.280845589828179*f_del*f_gam + 2.32979789038294*f_del*f_lav - 2.65793130908236*f_gam*f_lav + 3.03032773938404*f_lav*f_lav);
   return fict_gam_Cr_result;

}

__device__ double d_fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = -1.0*INV_DET*(1.1446875701682*XCR*f_del + 0.859772348881005*XCR*f_lav - 0.959509159362874*XNB*f_del - 16.4617409654023*XNB*f_gam - 4.01486311346988*XNB*f_lav + 0.213296730615793*f_del*f_del + 3.32436125411199*f_del*f_gam + 0.924657119203107*f_del*f_lav + 4.87295844646931*f_gam*f_lav + 1.04964871648403*f_lav*f_lav);
   return fict_gam_Nb_result;

}

__device__ double d_fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 1.0*INV_DET*(0.184507995023542*XCR*f_del + 0.959509159362873*XCR*f_gam + 0.124922254184693*XCR*f_lav + 2.30448864885622*XNB*f_gam - 0.168431241308811*XNB*f_lav - 0.566287612713812*f_del*f_gam - 0.00552028324855524*f_del*f_lav - 0.280845589828179*f_gam*f_gam - 0.977380717910862*f_gam*f_lav + 0.0233455789357588*f_lav*f_lav);
   return fict_del_Cr_result;

}

__device__ double d_fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = 1.0*INV_DET*(1.1446875701682*XCR*f_gam - 0.0162978336512057*XCR*f_lav + 0.184507995023542*XNB*f_del + 5.9147349211665*XNB*f_gam + 0.571097409189889*XNB*f_lav + 0.213296730615793*f_del*f_gam - 0.0296682025877037*f_del*f_lav + 3.32436125411199*f_gam*f_gam - 0.275088794670308*f_gam*f_lav - 0.091342735434936*f_lav*f_lav);
   return fict_del_Nb_result;

}

__device__ double d_fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 1.0*INV_DET*(0.571097409189889*XCR*f_del + 4.01486311346988*XCR*f_gam + 0.371787202791778*XCR*f_lav + 0.168431241308811*XNB*f_del + 12.5323221975227*XNB*f_gam + 0.00552028324855524*f_del*f_del - 1.35241717247208*f_del*f_gam - 0.0233455789357588*f_del*f_lav + 2.65793130908236*f_gam*f_gam - 3.03032773938404*f_gam*f_lav);
   return fict_lav_Cr_result;

}

__device__ double d_fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = 1.0*INV_DET*(0.0162978336512057*XCR*f_del + 0.859772348881005*XCR*f_gam + 0.124922254184693*XNB*f_del + 4.20816556923794*XNB*f_gam + 0.371787202791778*XNB*f_lav + 0.0296682025877037*f_del*f_del + 1.19974591387341*f_del*f_gam + 0.091342735434936*f_del*f_lav + 4.87295844646931*f_gam*f_gam + 1.04964871648403*f_gam*f_lav);
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
   g_gam_result = 2474387391.41825*(XCR - 0.49)*(XCR - 0.49) + (15148919424.3273*XCR - 7422970517.92036)*(XNB - 0.025) + 37770442067.0002*(XNB - 0.025)*(XNB - 0.025);
   return g_gam_result;

}

__device__ double d_g_del(double XCR, double XNB) {

   double g_del_result;
   g_del_result = 32328981887.7551*(XCR - 0.015)*(XCR - 0.015) + (16970167763.7333*XCR - 254552516.456)*(XNB - 0.245) + 101815797663.265*(XNB - 0.245)*(XNB - 0.245);
   return g_del_result;

}

__device__ double d_g_lav(double XCR, double XNB) {

   double g_lav_result;
   g_lav_result = (-3224431198.33485*XCR + 967329359.500456)*(XNB - 0.328) + 10490734425.5087*(XCR - 0.3)*(XCR - 0.3) + 152553892719.717*(XNB - 0.328)*(XNB - 0.328);
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
   dg_lav_dxCr_result = 20981468851.0174*XCR - 3224431198.33485*XNB - 5236827222.25137;
   return dg_lav_dxCr_result;

}

__device__ double d_dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = -3224431198.33485*XCR + 305107785439.433*XNB - 99108024264.6336;
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
   d2g_lav_dxCrNb_result = -3224431198.33485;
   return d2g_lav_dxCrNb_result;

}

__device__ double d_d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = -3224431198.33485;
   return d2g_lav_dxNbCr_result;

}

__device__ double d_d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 305107785439.433;
   return d2g_lav_dxNbNb_result;

}
