/******************************************************************************
 *                      Code generated with sympy 1.1.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
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

double xr_gam_Cr(double P_del, double P_lav) {

   double xr_gam_Cr_result;
   xr_gam_Cr_result = -1.35745662070089e-9*P_del + 6.39244600941496e-10*P_lav + 0.49;
   return xr_gam_Cr_result;

}

double xr_gam_Nb(double P_del, double P_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 2.47594876167461e-10*P_del - 6.66222300558977e-11*P_lav + 0.025;
   return xr_gam_Nb_result;

}

double xr_del_Cr(double P_del, double P_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = -4.44614260967702e-11*P_del + 2.79333077660349e-11*P_lav + 0.015;
   return xr_del_Cr_result;

}

double xr_del_Nb(double P_del, double P_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = -5.43111816527598e-12*P_del + 2.05131716667735e-11*P_lav + 0.245;
   return xr_del_Nb_result;

}

double xr_lav_Cr(double P_del, double P_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = -1.42576885211229e-10*P_del + 1.05186220574482e-10*P_lav + 0.3;
   return xr_lav_Cr_result;

}

double xr_lav_Nb(double P_del, double P_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = -7.60450759974508e-12*P_del + 1.63559527488478e-11*P_lav + 0.328;
   return xr_lav_Nb_result;

}

double fict_gam_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = 2.0*(5.18019310119937e+16*XCR*f_del + 1.4417382709332e+17*XCR*f_gam + 3.68555996862356e+16*XCR*f_lav - 2.01829775293512e+16*XNB*f_del - 1.09759524061327e+17*XNB*f_lav + 4.9596122628877e+15*pow(f_del, 2) + 2.45967808586641e+15*f_del*f_gam + 2.040463878738e+16*f_del*f_lav - 2.32784691356123e+16*f_gam*f_lav + 2.65399600474984e+16*pow(f_lav, 2))/(3.23188462602675e+15*pow(f_del, 2) + 1.20410845917997e+17*f_del*f_gam + 1.21916410678335e+16*f_del*f_lav + 2.88347654186639e+17*pow(f_gam, 2) + 1.44036468314717e+17*f_gam*f_lav + 6.51231045409673e+15*pow(f_lav, 2));
   return fict_gam_Cr_result;

}

double fict_gam_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = -2.0*(1.00253058388026e+16*XCR*f_del + 7.52998545097433e+15*XCR*f_lav - 8.40349194700494e+15*XNB*f_del - 1.4417382709332e+17*XNB*f_gam - 3.51626344711231e+16*XNB*f_lav + 1.86807738160884e+15*pow(f_del, 2) + 2.91151395015503e+16*f_del*f_gam + 8.09825375728952e+15*f_del*f_lav + 4.26779324234755e+16*f_gam*f_lav + 9.19294459055988e+15*pow(f_lav, 2))/(3.23188462602675e+15*pow(f_del, 2) + 1.20410845917997e+17*f_del*f_gam + 1.21916410678335e+16*f_del*f_lav + 2.88347654186639e+17*pow(f_gam, 2) + 1.44036468314717e+17*f_gam*f_lav + 6.51231045409673e+15*pow(f_lav, 2));
   return fict_gam_Nb_result;

}

double fict_del_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 0.03125*(1.03420308032856e+17*XCR*f_del + 5.37823484608316e+17*XCR*f_gam + 7.00213451796019e+16*XCR*f_lav + 1.29171056187848e+18*XNB*f_gam - 9.44089759161437e+16*XNB*f_lav - 3.17415184824813e+17*f_del*f_gam - 3.09422577553543e+15*f_del*f_lav - 1.5741939749545e+17*pow(f_gam, 2) - 5.47840839627672e+17*f_gam*f_lav + 1.3085649564581e+16*pow(f_lav, 2))/(3.23188462602675e+15*pow(f_del, 2) + 1.20410845917997e+17*f_del*f_gam + 1.21916410678335e+16*f_del*f_lav + 2.88347654186639e+17*pow(f_gam, 2) + 1.44036468314717e+17*f_gam*f_lav + 6.51231045409673e+15*pow(f_lav, 2));
   return fict_del_Cr_result;

}

double fict_del_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = 0.0625*(3.20809786841682e+17*XCR*f_gam - 4.56762585345118e+15*XCR*f_lav + 5.1710154016428e+16*XNB*f_del + 1.6576617923838e+18*XNB*f_gam + 1.60055584495535e+17*XNB*f_lav + 5.9778476211483e+16*f_del*f_gam - 8.31480134508535e+15*f_del*f_lav + 9.3168446404961e+17*pow(f_gam, 2) - 7.70963011048935e+16*f_gam*f_lav - 2.55996869784408e+16*pow(f_lav, 2))/(3.23188462602675e+15*pow(f_del, 2) + 1.20410845917997e+17*f_del*f_gam + 1.21916410678335e+16*f_del*f_lav + 2.88347654186639e+17*pow(f_gam, 2) + 1.44036468314717e+17*f_gam*f_lav + 6.51231045409673e+15*pow(f_lav, 2));
   return fict_del_Nb_result;

}

double fict_lav_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 0.03125*(3.20111168991069e+17*XCR*f_del + 2.25040860615188e+18*XCR*f_gam + 2.08393934531095e+17*XCR*f_lav + 9.44089759161437e+16*XNB*f_del + 7.02460953992495e+18*XNB*f_gam + 3.09422577553543e+15*pow(f_del, 2) - 7.58056042764651e+17*f_del*f_gam - 1.3085649564581e+16*f_del*f_lav + 1.48982202467919e+18*pow(f_gam, 2) - 1.6985574430399e+18*f_gam*f_lav)/(3.23188462602675e+15*pow(f_del, 2) + 1.20410845917997e+17*f_del*f_gam + 1.21916410678335e+16*f_del*f_lav + 2.88347654186639e+17*pow(f_gam, 2) + 1.44036468314717e+17*f_gam*f_lav + 6.51231045409673e+15*pow(f_lav, 2));
   return fict_lav_Cr_result;

}

double fict_lav_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = 0.0625*(4.56762585345118e+15*XCR*f_del + 2.40959534431179e+17*XCR*f_gam + 3.5010672589801e+16*XNB*f_del + 1.17937918995954e+18*XNB*f_gam + 1.04196967265548e+17*XNB*f_lav + 8.31480134508535e+15*pow(f_del, 2) + 3.36240421338158e+17*f_del*f_gam + 2.55996869784408e+16*f_del*f_lav + 1.36569383755122e+18*pow(f_gam, 2) + 2.94174226897916e+17*f_gam*f_lav)/(3.23188462602675e+15*pow(f_del, 2) + 1.20410845917997e+17*f_del*f_gam + 1.21916410678335e+16*f_del*f_lav + 2.88347654186639e+17*pow(f_gam, 2) + 1.44036468314717e+17*f_gam*f_lav + 6.51231045409673e+15*pow(f_lav, 2));
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
   g_lav_result = (-3224431198.33484*XCR + 967329359.500452)*(XNB - 0.328) + 10490734425.5087*pow(XCR - 0.3, 2) + 152553892719.717*pow(XNB - 0.328, 2);
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
   dg_lav_dxCr_result = 20981468851.0174*XCR - 3224431198.33484*XNB - 5236827222.25138;
   return dg_lav_dxCr_result;

}

double dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = -3224431198.33484*XCR + 305107785439.433*XNB - 99108024264.6336;
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
   d2g_lav_dxCrNb_result = -3224431198.33484;
   return d2g_lav_dxCrNb_result;

}

double d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = -3224431198.33484;
   return d2g_lav_dxNbCr_result;

}

double d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 305107785439.433;
   return d2g_lav_dxNbNb_result;

}
