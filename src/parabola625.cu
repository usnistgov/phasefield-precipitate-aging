/******************************************************************************
 *                       Code generated with sympy 1.3                        *
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

__device__ double d_interface_profile(double L, double r) {

   double interface_profile_result;
   interface_profile_result = -0.5*tanh(r/L) + 0.5;
   return interface_profile_result;

}

__device__ double d_kT() {

   double kT_result;
   kT_result = 1.5782883556379999e-20;
   return kT_result;

}

__device__ double d_xe_gam_Cr() {

   double xe_gam_Cr_result;
   xe_gam_Cr_result = 0.490;
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
   xe_lav_Cr_result = 0.300;
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
   xr_gam_Cr_result = -1.3574566207013822e-9*P_del + 6.3924460094164553e-10*P_lav + 0.490;
   return xr_gam_Cr_result;

}

__device__ double d_xr_gam_Nb(double P_del, double P_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 2.4759487616754483e-10*P_del - 6.6622230055923671e-11*P_lav + 0.025;
   return xr_gam_Nb_result;

}

__device__ double d_xr_del_Cr(double P_del, double P_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = -4.4461426096786933e-11*P_del + 2.793330776604002e-11*P_lav + 0.015;
   return xr_del_Cr_result;

}

__device__ double d_xr_del_Nb(double P_del, double P_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = -5.4311181652802693e-12*P_del + 2.0513171666774676e-11*P_lav + 0.245;
   return xr_del_Nb_result;

}

__device__ double d_xr_lav_Cr(double P_del, double P_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = -1.4257688521128304e-10*P_del + 1.0518622057449897e-10*P_lav + 0.300;
   return xr_lav_Cr_result;

}

__device__ double d_xr_lav_Nb(double P_del, double P_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = -7.6045075997491245e-12*P_del + 1.6355952748848998e-11*P_lav + 0.328;
   return xr_lav_Nb_result;

}

__device__ double d_inv_fict_det(double f_del, double f_gam, double f_lav) {

   double inv_fict_det_result;
   inv_fict_det_result = 0.060746916264914186/(0.011208291723902289*f_del*f_del + 0.41758912954449995*f_del*f_gam + 0.04228104820974949*f_del*f_lav + 1.0*f_gam*f_gam + 0.49952363483243967*f_gam*f_lav + 0.022584926076358817*f_lav*f_lav);
   return inv_fict_det_result;

}

__device__ double d_fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = 16.461740965402281*INV_DET*(0.35930190698526532*XCR*f_del + 1.0*XCR*f_gam + 0.25563308146339203*XCR*f_lav - 0.13999057898551379*XNB*f_del - 0.76129992713783667*XNB*f_lav + 0.034400226191384116*f_del*f_del + 0.017060503528662844*f_del*f_gam + 0.14152803736126604*f_del*f_lav - 0.16146113067071988*f_gam*f_lav + 0.18408306543960862*f_lav*f_lav);
   return fict_gam_Cr_result;

}

__device__ double d_fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = -16.461740965402281*INV_DET*(0.0695362399744959*XCR*f_del + 0.052228518884363015*XCR*f_lav - 0.058287222559234672*XNB*f_del - 1.0*XNB*f_gam - 0.24389055336904775*XNB*f_lav + 0.012957118634297542*f_del*f_del + 0.20194469473786605*f_del*f_gam + 0.056170068593987918*f_del*f_lav + 0.29601719871007703*f_gam*f_lav + 0.063762922687829737*f_lav*f_lav);
   return fict_gam_Nb_result;

}

__device__ double d_fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 2.304488648856216*INV_DET*(0.080064614384244545*XCR*f_del + 0.41636532244977881*XCR*f_gam + 0.05420823150797275*XCR*f_lav + 1.0*XNB*f_gam - 0.073088336274690788*XNB*f_lav - 0.24573243743026343*f_del*f_gam - 0.0023954482272217311*f_del*f_lav - 0.12186894041225636*f_gam*f_gam - 0.42412043053280546*f_gam*f_lav + 0.010130481201261664*f_lav*f_lav);
   return fict_del_Cr_result;

}

__device__ double d_fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = 5.9147349211665023*INV_DET*(0.19353150824592621*XCR*f_gam - 0.0027554630711990537*XCR*f_lav + 0.031194634667947731*XNB*f_del + 1.0*XNB*f_gam + 0.096555030242548426*XNB*f_lav + 0.036061925590695244*f_del*f_gam - 0.005015981778242141*f_del*f_lav + 0.56204737801780613*f_gam*f_gam - 0.046509065636377266*f_gam*f_lav - 0.015443250906825292*f_lav*f_lav);
   return fict_del_Nb_result;

}

__device__ double d_fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 12.532322197522699*INV_DET*(0.045569959037821371*XCR*f_del + 0.32036066821386966*XCR*f_gam + 0.029666265910819816*XCR*f_lav + 0.013439747131788969*XNB*f_del + 1.0*XNB*f_gam + 0.00044048366787493193*f_del*f_del - 0.10791433153062506*f_del*f_gam - 0.0018628294555316793*f_del*f_lav + 0.21208609762744221*f_gam*f_gam - 0.24180097603802814*f_gam*f_lav);
   return fict_lav_Cr_result;

}

__device__ double d_fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = 4.8729584464693021*INV_DET*(0.0033445459940283965*XCR*f_del + 0.17643744725628691*XCR*f_gam + 0.025635813552895237*XNB*f_del + 0.86357509826232159*XNB*f_gam + 0.076295992850330216*XNB*f_lav + 0.0060883348203389078*f_del*f_del + 0.24620483163419699*f_del*f_gam + 0.018744821331509221*f_del*f_lav + 1.0*f_gam*f_gam + 0.21540276364239214*f_gam*f_lav);
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
   s_delta_result = 1.010;
   return s_delta_result;

}

__device__ double d_s_laves() {

   double s_laves_result;
   s_laves_result = 1.011;
   return s_laves_result;

}

__device__ double d_g_gam(double XCR, double XNB) {

   double g_gam_result;
   g_gam_result = 2474387391.4182534*pow(XCR - 0.490, 2) + (15148919424.327274*XCR - 7422970517.9203644)*(XNB - 0.025) + 37770442067.000183*pow(XNB - 0.025, 2);
   return g_gam_result;

}

__device__ double d_g_del(double XCR, double XNB) {

   double g_del_result;
   g_del_result = 32328981887.7551*pow(XCR - 0.015, 2) + (16970167763.733316*XCR - 254552516.45599973)*(XNB - 0.245) + 101815797663.26523*(XNB - 0.245)*(XNB - 0.245);
   return g_del_result;

}

__device__ double d_g_lav(double XCR, double XNB) {

   double g_lav_result;
   g_lav_result = (-3224431198.3348541*XCR + 967329359.50045621)*(XNB - 0.328) + 10490734425.508677*pow(XCR - 0.300, 2) + 152553892719.7164*pow(XNB - 0.328, 2);
   return g_lav_result;

}

__device__ double d_dg_gam_dxCr(double XCR, double XNB) {

   double dg_gam_dxCr_result;
   dg_gam_dxCr_result = 4948774782.8365068*XCR + 15148919424.327274*XNB - 2803622629.19807;
   return dg_gam_dxCr_result;

}

__device__ double d_dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 15148919424.327274*XCR + 75540884134.000366*XNB - 9311492621.2703743;
   return dg_gam_dxNb_result;

}

__device__ double d_dg_del_dxCr(double XCR, double XNB) {

   double dg_del_dxCr_result;
   dg_del_dxCr_result = 64657963775.510201*XCR + 16970167763.733316*XNB - 5127560558.7473154;
   return dg_del_dxCr_result;

}

__device__ double d_dg_del_dxNb(double XCR, double XNB) {

   double dg_del_dxNb_result;
   dg_del_dxNb_result = 16970167763.733316*XCR + 203631595326.53046*XNB - 50144293371.455963;
   return dg_del_dxNb_result;

}

__device__ double d_dg_lav_dxCr(double XCR, double XNB) {

   double dg_lav_dxCr_result;
   dg_lav_dxCr_result = 20981468851.017353*XCR - 3224431198.3348541*XNB - 5236827222.2513733;
   return dg_lav_dxCr_result;

}

__device__ double d_dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = -3224431198.3348541*XCR + 305107785439.4328*XNB - 99108024264.633499;
   return dg_lav_dxNb_result;

}

__device__ double d_d2g_gam_dxCrCr() {

   double d2g_gam_dxCrCr_result;
   d2g_gam_dxCrCr_result = 4948774782.8365068;
   return d2g_gam_dxCrCr_result;

}

__device__ double d_d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 15148919424.327274;
   return d2g_gam_dxCrNb_result;

}

__device__ double d_d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 15148919424.327274;
   return d2g_gam_dxNbCr_result;

}

__device__ double d_d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 75540884134.000366;
   return d2g_gam_dxNbNb_result;

}

__device__ double d_d2g_del_dxCrCr() {

   double d2g_del_dxCrCr_result;
   d2g_del_dxCrCr_result = 64657963775.510201;
   return d2g_del_dxCrCr_result;

}

__device__ double d_d2g_del_dxCrNb() {

   double d2g_del_dxCrNb_result;
   d2g_del_dxCrNb_result = 16970167763.733316;
   return d2g_del_dxCrNb_result;

}

__device__ double d_d2g_del_dxNbCr() {

   double d2g_del_dxNbCr_result;
   d2g_del_dxNbCr_result = 16970167763.733316;
   return d2g_del_dxNbCr_result;

}

__device__ double d_d2g_del_dxNbNb() {

   double d2g_del_dxNbNb_result;
   d2g_del_dxNbNb_result = 203631595326.53046;
   return d2g_del_dxNbNb_result;

}

__device__ double d_d2g_lav_dxCrCr() {

   double d2g_lav_dxCrCr_result;
   d2g_lav_dxCrCr_result = 20981468851.017353;
   return d2g_lav_dxCrCr_result;

}

__device__ double d_d2g_lav_dxCrNb() {

   double d2g_lav_dxCrNb_result;
   d2g_lav_dxCrNb_result = -3224431198.3348541;
   return d2g_lav_dxCrNb_result;

}

__device__ double d_d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = -3224431198.3348541;
   return d2g_lav_dxNbCr_result;

}

__device__ double d_d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 305107785439.4328;
   return d2g_lav_dxNbNb_result;

}
