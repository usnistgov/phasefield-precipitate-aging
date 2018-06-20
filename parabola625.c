/******************************************************************************
 *                      Code generated with sympy 1.1.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *            This file is part of 'phasefield-precipitate-aging'             *
 ******************************************************************************/
#include "parabola625.h"
#include <math.h>

double xe_gam_Cr() {

   double xe_gam_Cr_result;
   xe_gam_Cr_result = 0.49;
   return xe_gam_Cr_result;

}

double xe_gam_Nb() {

   double xe_gam_Nb_result;
   xe_gam_Nb_result = 0.025;
   return xe_gam_Nb_result;

}

double xe_del_Cr() {

   double xe_del_Cr_result;
   xe_del_Cr_result = 0.015;
   return xe_del_Cr_result;

}

double xe_del_Nb() {

   double xe_del_Nb_result;
   xe_del_Nb_result = 0.245;
   return xe_del_Nb_result;

}

double xe_lav_Cr() {

   double xe_lav_Cr_result;
   xe_lav_Cr_result = 0.3;
   return xe_lav_Cr_result;

}

double xe_lav_Nb() {

   double xe_lav_Nb_result;
   xe_lav_Nb_result = 0.328;
   return xe_lav_Nb_result;

}

double r_delta() {

   double r_delta_result;
   r_delta_result = 7.5e-9;
   return r_delta_result;

}

double r_laves() {

   double r_laves_result;
   r_laves_result = 7.5e-9;
   return r_laves_result;

}

double s_delta() {

   double s_delta_result;
   s_delta_result = 1.01;
   return s_delta_result;

}

double s_laves() {

   double s_laves_result;
   s_laves_result = 1.011;
   return s_laves_result;

}

double g_gam(double XCR, double XNB) {

   double g_gam_result;
   g_gam_result = 2474387391.41825*pow(XCR - 0.49, 2) + (15148919424.3273*XCR - 7422970517.92036)*(XNB - 0.025) + 37770442067.0002*pow(XNB - 0.025, 2);
   return g_gam_result;

}

double g_del(double XCR, double XNB) {

   double g_del_result;
   g_del_result = 32328981887.7551*pow(XCR - 0.015, 2) + (16970167763.7333*XCR - 254552516.456)*(XNB - 0.245) + 101815797663.265*pow(XNB - 0.245, 2);
   return g_del_result;

}

double g_lav(double XCR, double XNB) {

   double g_lav_result;
   g_lav_result = (-3224431198.33485*XCR + 967329359.500456)*(XNB - 0.328) + 10490734425.5087*pow(XCR - 0.3, 2) + 152553892719.717*pow(XNB - 0.328, 2);
   return g_lav_result;

}

double dg_gam_dxCr(double XCR, double XNB) {

   double dg_gam_dxCr_result;
   dg_gam_dxCr_result = 4948774782.83651*XCR + 15148919424.3273*XNB - 2803622629.19807;
   return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 15148919424.3273*XCR + 75540884134.0004*XNB - 9311492621.27037;
   return dg_gam_dxNb_result;

}

double dg_del_dxCr(double XCR, double XNB) {

   double dg_del_dxCr_result;
   dg_del_dxCr_result = 64657963775.5102*XCR + 16970167763.7333*XNB - 5127560558.74732;
   return dg_del_dxCr_result;

}

double dg_del_dxNb(double XCR, double XNB) {

   double dg_del_dxNb_result;
   dg_del_dxNb_result = 16970167763.7333*XCR + 203631595326.53*XNB - 50144293371.456;
   return dg_del_dxNb_result;

}

double dg_lav_dxCr(double XCR, double XNB) {

   double dg_lav_dxCr_result;
   dg_lav_dxCr_result = 20981468851.0174*XCR - 3224431198.33485*XNB - 5236827222.25137;
   return dg_lav_dxCr_result;

}

double dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = -3224431198.33485*XCR + 305107785439.433*XNB - 99108024264.6336;
   return dg_lav_dxNb_result;

}

double d2g_gam_dxCrCr() {

   double d2g_gam_dxCrCr_result;
   d2g_gam_dxCrCr_result = 4948774782.83651;
   return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 15148919424.3273;
   return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 15148919424.3273;
   return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 75540884134.0004;
   return d2g_gam_dxNbNb_result;

}

double d2g_del_dxCrCr() {

   double d2g_del_dxCrCr_result;
   d2g_del_dxCrCr_result = 64657963775.5102;
   return d2g_del_dxCrCr_result;

}

double d2g_del_dxCrNb() {

   double d2g_del_dxCrNb_result;
   d2g_del_dxCrNb_result = 16970167763.7333;
   return d2g_del_dxCrNb_result;

}

double d2g_del_dxNbCr() {

   double d2g_del_dxNbCr_result;
   d2g_del_dxNbCr_result = 16970167763.7333;
   return d2g_del_dxNbCr_result;

}

double d2g_del_dxNbNb() {

   double d2g_del_dxNbNb_result;
   d2g_del_dxNbNb_result = 203631595326.53;
   return d2g_del_dxNbNb_result;

}

double d2g_lav_dxCrCr() {

   double d2g_lav_dxCrCr_result;
   d2g_lav_dxCrCr_result = 20981468851.0174;
   return d2g_lav_dxCrCr_result;

}

double d2g_lav_dxCrNb() {

   double d2g_lav_dxCrNb_result;
   d2g_lav_dxCrNb_result = -3224431198.33485;
   return d2g_lav_dxCrNb_result;

}

double d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = -3224431198.33485;
   return d2g_lav_dxNbCr_result;

}

double d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 305107785439.433;
   return d2g_lav_dxNbNb_result;

}
