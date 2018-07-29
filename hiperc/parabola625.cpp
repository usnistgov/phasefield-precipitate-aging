/******************************************************************************
 *                      Code generated with sympy 1.1.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/
#include "parabola625.h"
#include <math.h>

double h(double x) {

   double h_result;
   h_result = pow(x, 3)*(6.0*pow(x, 2) - 15.0*x + 10.0);
   return h_result;

}

double hprime(double x) {

   double hprime_result;
   hprime_result = 30.0*pow(x, 2)*pow(-x + 1.0, 2);
   return hprime_result;

}

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

double Vm() {

   double Vm_result;
   Vm_result = 1.0e-5;
   return Vm_result;

}

double xr_gam_Cr(double P_del, double P_lav) {

   double xr_gam_Cr_result;
   xr_gam_Cr_result = -1.35745662070135e-9*P_del + 6.39244600941618e-10*P_lav + 0.49;
   return xr_gam_Cr_result;

}

double xr_gam_Nb(double P_del, double P_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 2.4759487616754e-10*P_del - 6.6622230055919e-11*P_lav + 0.025;
   return xr_gam_Nb_result;

}

double xr_del_Cr(double P_del, double P_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = -4.44614260967858e-11*P_del + 2.79333077660391e-11*P_lav + 0.015;
   return xr_del_Cr_result;

}

double xr_del_Nb(double P_del, double P_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = -5.4311181652798e-12*P_del + 2.05131716667744e-11*P_lav + 0.245;
   return xr_del_Nb_result;

}

double xr_lav_Cr(double P_del, double P_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = -1.42576885211279e-10*P_del + 1.05186220574496e-10*P_lav + 0.3;
   return xr_lav_Cr_result;

}

double xr_lav_Nb(double P_del, double P_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = -7.60450759974872e-12*P_del + 1.63559527488487e-11*P_lav + 0.328;
   return xr_lav_Nb_result;

}

double fict_gam_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = 2.0*(6.63064716953519e+16*XCR*f_del + 1.84542498679449e+17*XCR*f_gam + 4.71751675983815e+16*XCR*f_lav - 2.58342112375695e+16*XNB*f_del - 1.40492190798499e+17*XNB*f_lav + 6.34830369649626e+15*pow(f_del, 2) + 3.148387949909e+15*f_del*f_gam + 2.61179376478465e+16*f_del*f_lav - 2.97964404935837e+16*f_gam*f_lav + 3.3971148860798e+16*pow(f_lav, 2))/(4.13681232131424e+15*pow(f_del, 2) + 1.54125882775036e+17*f_del*f_gam + 1.56053005668268e+16*f_del*f_lav + 3.69084997358898e+17*pow(f_gam, 2) + 1.84366679442838e+17*f_gam*f_lav + 8.33575738124381e+15*pow(f_lav, 2));
   return fict_gam_Cr_result;

}

double fict_gam_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = -4.0*(6.41619573683364e+15*XCR*f_del + 4.81919068862358e+15*XCR*f_lav - 5.37823484608316e+15*XNB*f_del - 9.22712493397246e+16*XNB*f_gam - 2.25040860615188e+16*XNB*f_lav + 1.19556952422966e+15*pow(f_del, 2) + 1.86336892809922e+16*f_del*f_gam + 5.18288240466529e+15*f_del*f_lav + 2.73138767510243e+16*f_gam*f_lav + 5.88348453795833e+15*pow(f_lav, 2))/(4.13681232131424e+15*pow(f_del, 2) + 1.54125882775036e+17*f_del*f_gam + 1.56053005668268e+16*f_del*f_lav + 3.69084997358898e+17*pow(f_gam, 2) + 1.84366679442838e+17*f_gam*f_lav + 8.33575738124381e+15*pow(f_lav, 2));
   return fict_gam_Nb_result;

}

double fict_del_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 0.015625*(2.64755988564111e+17*XCR*f_del + 1.37682812059729e+18*XCR*f_gam + 1.79254643659781e+17*XCR*f_lav + 3.3067790384089e+18*XNB*f_gam - 2.41686978345328e+17*XNB*f_lav - 8.12582873151521e+17*f_del*f_gam - 7.92121798537059e+15*f_del*f_lav - 4.02993657588352e+17*pow(f_gam, 2) - 1.40247254944684e+18*f_gam*f_lav + 3.34992628853275e+16*pow(f_lav, 2))/(4.13681232131424e+15*pow(f_del, 2) + 1.54125882775036e+17*f_del*f_gam + 1.56053005668268e+16*f_del*f_lav + 3.69084997358898e+17*pow(f_gam, 2) + 1.84366679442838e+17*f_gam*f_lav + 8.33575738124381e+15*pow(f_lav, 2));
   return fict_del_Cr_result;

}

double fict_del_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = 0.0625*(4.10636527157353e+17*XCR*f_gam - 5.84656109241752e+15*XCR*f_lav + 6.61889971410278e+16*XNB*f_del + 2.12180709425126e+18*XNB*f_gam + 2.04871148154284e+17*XNB*f_lav + 7.65164495506982e+16*f_del*f_gam - 1.06429457217093e+16*f_del*f_lav + 1.1925561139835e+18*pow(f_gam, 2) - 9.86832654142635e+16*f_gam*f_lav - 3.27675993324042e+16*pow(f_lav, 2))/(4.13681232131424e+15*pow(f_del, 2) + 1.54125882775036e+17*f_del*f_gam + 1.56053005668268e+16*f_del*f_lav + 3.69084997358898e+17*pow(f_gam, 2) + 1.84366679442838e+17*f_gam*f_lav + 8.33575738124381e+15*pow(f_lav, 2));
   return fict_del_Nb_result;

}

double fict_lav_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 0.015625*(8.19484592617137e+17*XCR*f_del + 5.76104603174881e+18*XCR*f_gam + 5.33488472399604e+17*XCR*f_lav + 2.41686978345328e+17*XNB*f_del + 1.79830004222079e+19*XNB*f_gam + 7.92121798537059e+15*pow(f_del, 2) - 1.94062346947751e+18*f_del*f_gam - 3.34992628853275e+16*f_del*f_lav + 3.81394438317871e+18*pow(f_gam, 2) - 4.34830705418214e+18*f_gam*f_lav)/(4.13681232131424e+15*pow(f_del, 2) + 1.54125882775036e+17*f_del*f_gam + 1.56053005668268e+16*f_del*f_lav + 3.69084997358898e+17*pow(f_gam, 2) + 1.84366679442838e+17*f_gam*f_lav + 8.33575738124381e+15*pow(f_lav, 2));
   return fict_lav_Cr_result;

}

double fict_lav_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = 0.0625*(5.84656109241752e+15*XCR*f_del + 3.08428204071909e+17*XCR*f_gam + 4.48136609149452e+16*XNB*f_del + 1.50960536314821e+18*XNB*f_gam + 1.33372118099901e+17*XNB*f_lav + 1.06429457217093e+16*pow(f_del, 2) + 4.30387739312842e+17*f_del*f_gam + 3.27675993324042e+16*f_del*f_lav + 1.74808811206556e+18*pow(f_gam, 2) + 3.76543010429333e+17*f_gam*f_lav)/(4.13681232131424e+15*pow(f_del, 2) + 1.54125882775036e+17*f_del*f_gam + 1.56053005668268e+16*f_del*f_lav + 3.69084997358898e+17*pow(f_gam, 2) + 1.84366679442838e+17*f_gam*f_lav + 8.33575738124381e+15*pow(f_lav, 2));
   return fict_lav_Nb_result;

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
