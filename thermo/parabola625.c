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

double interface_profile(double z) {

   double interface_profile_result;
   interface_profile_result = -1.0L/2.0L*tanh(z) + 1.0L/2.0L;
   return interface_profile_result;

}

double kT() {

   double kT_result;
   kT_result = 1.578288355638e-20;
   return kT_result;

}

double RT() {

   double RT_result;
   RT_result = 9504.6886668;
   return RT_result;

}

double Vm() {

   double Vm_result;
   Vm_result = 1.0e-5;
   return Vm_result;

}

double xe_gam_Cr() {

   double xe_gam_Cr_result;
   xe_gam_Cr_result = 0.525;
   return xe_gam_Cr_result;

}

double xe_gam_Nb() {

   double xe_gam_Nb_result;
   xe_gam_Nb_result = 0.018;
   return xe_gam_Nb_result;

}

double xe_del_Cr() {

   double xe_del_Cr_result;
   xe_del_Cr_result = 0.0258;
   return xe_del_Cr_result;

}

double xe_del_Nb() {

   double xe_del_Nb_result;
   xe_del_Nb_result = 0.244;
   return xe_del_Nb_result;

}

double xe_lav_Cr() {

   double xe_lav_Cr_result;
   xe_lav_Cr_result = 0.375;
   return xe_lav_Cr_result;

}

double xe_lav_Nb() {

   double xe_lav_Nb_result;
   xe_lav_Nb_result = 0.259;
   return xe_lav_Nb_result;

}

double matrix_min_Cr() {

   double matrix_min_Cr_result;
   matrix_min_Cr_result = 0.2794;
   return matrix_min_Cr_result;

}

double matrix_max_Cr() {

   double matrix_max_Cr_result;
   matrix_max_Cr_result = 0.3288;
   return matrix_max_Cr_result;

}

double matrix_min_Nb() {

   double matrix_min_Nb_result;
   matrix_min_Nb_result = 0.0202;
   return matrix_min_Nb_result;

}

double matrix_max_Nb() {

   double matrix_max_Nb_result;
   matrix_max_Nb_result = 0.0269;
   return matrix_max_Nb_result;

}

double enrich_min_Cr() {

   double enrich_min_Cr_result;
   enrich_min_Cr_result = 0.2473;
   return enrich_min_Cr_result;

}

double enrich_max_Cr() {

   double enrich_max_Cr_result;
   enrich_max_Cr_result = 0.2967;
   return enrich_max_Cr_result;

}

double enrich_min_Nb() {

   double enrich_min_Nb_result;
   enrich_min_Nb_result = 0.1659;
   return enrich_min_Nb_result;

}

double enrich_max_Nb() {

   double enrich_max_Nb_result;
   enrich_max_Nb_result = 0.1726;
   return enrich_max_Nb_result;

}

double xr_gam_Cr(double P_del, double P_lav) {

   double xr_gam_Cr_result;
   xr_gam_Cr_result = -1.26787888311728e-9*P_del + 8.39033790614187e-10*P_lav + 0.525;
   return xr_gam_Cr_result;

}

double xr_gam_Nb(double P_del, double P_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 1.91495394671797e-10*P_del - 7.50734464912494e-11*P_lav + 0.018;
   return xr_gam_Nb_result;

}

double xr_del_Cr(double P_del, double P_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = -7.17647843094884e-11*P_del + 5.60535326058406e-11*P_lav + 0.0258;
   return xr_del_Cr_result;

}

double xr_del_Nb(double P_del, double P_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = -3.00985604766955e-12*P_del + 2.80830668995081e-11*P_lav + 0.244;
   return xr_del_Nb_result;

}

double xr_lav_Cr(double P_del, double P_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = -1.73839015831777e-10*P_del + 1.2866636929017e-10*P_lav + 0.375;
   return xr_lav_Cr_result;

}

double xr_lav_Nb(double P_del, double P_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = 1.24465535321084e-11*P_del + 1.37713537054354e-11*P_lav + 0.259;
   return xr_lav_Nb_result;

}

double inv_fict_det(double f_del, double f_gam, double f_lav) {

   double inv_fict_det_result;
   inv_fict_det_result = 1.0/(0.0255946225237097*pow(f_del, 2) + 0.532594716713253*f_del*f_gam + 0.0856632429767053*f_del*f_lav + 0.907646695825129*pow(f_gam, 2) + 0.547354031767842*f_gam*f_lav + 0.0553767075065077*pow(f_lav, 2));
   return inv_fict_det_result;

}

double fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = 1.0*INV_DET*(0.458494450036513*XCR*f_del + 0.907646695825129*XCR*f_gam + 0.386740596414641*XCR*f_lav - 0.150463639332262*XNB*f_del - 0.239452101616465*XNB*f_lav + 0.0383211480110774*pow(f_del, 2) + 0.0181937007609806*f_del*f_gam - 0.0395437282069472*f_del*f_lav - 0.251735319544897*f_gam*f_lav - 0.0539368578959098*pow(f_lav, 2));
   return fict_gam_Cr_result;

}

double fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = -1.0*INV_DET*(0.071403872062733*XCR*f_del + 0.0495015500249432*XCR*f_lav - 0.0741002666767392*XNB*f_del - 0.907646695825129*XNB*f_gam - 0.160613435353201*XNB*f_lav + 0.0157775419644791*pow(f_del, 2) + 0.175725860847739*f_del*f_gam + 0.0287861169077074*f_del*f_lav + 0.20213084972015*f_gam*f_lav + 0.0220390177620083*pow(f_lav, 2));
   return fict_gam_Nb_result;

}

double fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 1.0*INV_DET*(0.0255946225237097*XCR*f_del + 0.0741002666767392*XCR*f_gam + 0.0233674601111437*XCR*f_lav + 0.150463639332262*XNB*f_gam + 0.00707656123506832*XNB*f_lav - 0.0383211480110774*f_del*f_gam - 0.00971743318981633*f_del*f_lav - 0.0181937007609806*pow(f_gam, 2) - 0.0650312432318045*f_gam*f_lav - 0.00916690784789372*pow(f_lav, 2));
   return fict_del_Cr_result;

}

double fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = 1.0*INV_DET*(0.071403872062733*XCR*f_gam + 0.0054190580275769*XCR*f_lav + 0.0255946225237097*XNB*f_del + 0.458494450036513*XNB*f_gam + 0.0622957828655616*XNB*f_lav + 0.0157775419644791*f_del*f_gam - 0.00106715866363323*f_del*f_lav + 0.175725860847739*pow(f_gam, 2) - 0.0159384603876866*f_gam*f_lav - 0.00465483789093393*pow(f_lav, 2));
   return fict_del_Nb_result;

}

double fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 1.0*INV_DET*(0.0622957828655616*XCR*f_del + 0.160613435353201*XCR*f_gam + 0.0553767075065077*XCR*f_lav - 0.00707656123506832*XNB*f_del + 0.239452101616465*XNB*f_gam + 0.00971743318981633*pow(f_del, 2) + 0.104574971438752*f_del*f_gam + 0.00916690784789372*f_del*f_lav + 0.251735319544897*pow(f_gam, 2) + 0.0539368578959098*f_gam*f_lav);
   return fict_lav_Cr_result;

}

double fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = -1.0*INV_DET*(0.0054190580275769*XCR*f_del - 0.0495015500249432*XCR*f_gam - 0.0233674601111437*XNB*f_del - 0.386740596414641*XNB*f_gam - 0.0553767075065077*XNB*f_lav - 0.00106715866363323*pow(f_del, 2) - 0.044724577295394*f_del*f_gam - 0.00465483789093393*f_del*f_lav - 0.20213084972015*pow(f_gam, 2) - 0.0220390177620083*f_gam*f_lav);
   return fict_lav_Nb_result;

}

double s_delta() {

   double s_delta_result;
   s_delta_result = 0.222;
   return s_delta_result;

}

double s_laves() {

   double s_laves_result;
   s_laves_result = 0.2775;
   return s_laves_result;

}

double g_gam(double XCR, double XNB) {

   double g_gam_result;
   g_gam_result = 2225414588.90076*pow(XCR - 0.525, 2) + (14903715603.9874*XCR - 7824450692.09337)*(XNB - 0.018) + 44805620198.9839*pow(XNB - 0.018, 2);
   return g_gam_result;

}

double g_del(double XCR, double XNB) {

   double g_del_result;
   g_del_result = 19076223392.8537*pow(XCR - 0.0258, 2) + (16983307497.0667*XCR - 438169333.42432)*(XNB - 0.244) + 85911591727.9098*pow(XNB - 0.244, 2);
   return g_del_result;

}

double g_lav(double XCR, double XNB) {

   double g_lav_result;
   g_lav_result = 8880622842.36342*pow(XCR - 0.375, 2) + (23980778596.6151*XCR - 8992791973.73065)*(XNB - 0.259) + 97731010299.2155*pow(XNB - 0.259, 2);
   return g_lav_result;

}

double dg_gam_dxCr(double XCR, double XNB) {

   double dg_gam_dxCr_result;
   dg_gam_dxCr_result = 4450829177.80152*XCR + 14903715603.9874*XNB - 2604952199.21757;
   return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 14903715603.9874*XCR + 89611240397.9677*XNB - 9437453019.25679;
   return dg_gam_dxNb_result;

}

double dg_del_dxCr(double XCR, double XNB) {

   double dg_del_dxCr_result;
   dg_del_dxCr_result = 38152446785.7074*XCR + 16983307497.0667*XNB - 5128260156.35551;
   return dg_del_dxCr_result;

}

double dg_del_dxNb(double XCR, double XNB) {

   double dg_del_dxNb_result;
   dg_del_dxNb_result = 16983307497.0667*XCR + 171823183455.82*XNB - 42363026096.6443;
   return dg_del_dxNb_result;

}

double dg_lav_dxCr(double XCR, double XNB) {

   double dg_lav_dxCr_result;
   dg_lav_dxCr_result = 17761245684.7268*XCR + 23980778596.6151*XNB - 12871488788.2959;
   return dg_lav_dxCr_result;

}

double dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = 23980778596.6151*XCR + 195462020598.431*XNB - 59617455308.7243;
   return dg_lav_dxNb_result;

}

double d2g_gam_dxCrCr() {

   double d2g_gam_dxCrCr_result;
   d2g_gam_dxCrCr_result = 4450829177.80152;
   return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 14903715603.9874;
   return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 14903715603.9874;
   return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 89611240397.9677;
   return d2g_gam_dxNbNb_result;

}

double d2g_del_dxCrCr() {

   double d2g_del_dxCrCr_result;
   d2g_del_dxCrCr_result = 38152446785.7074;
   return d2g_del_dxCrCr_result;

}

double d2g_del_dxCrNb() {

   double d2g_del_dxCrNb_result;
   d2g_del_dxCrNb_result = 16983307497.0667;
   return d2g_del_dxCrNb_result;

}

double d2g_del_dxNbCr() {

   double d2g_del_dxNbCr_result;
   d2g_del_dxNbCr_result = 16983307497.0667;
   return d2g_del_dxNbCr_result;

}

double d2g_del_dxNbNb() {

   double d2g_del_dxNbNb_result;
   d2g_del_dxNbNb_result = 171823183455.82;
   return d2g_del_dxNbNb_result;

}

double d2g_lav_dxCrCr() {

   double d2g_lav_dxCrCr_result;
   d2g_lav_dxCrCr_result = 17761245684.7268;
   return d2g_lav_dxCrCr_result;

}

double d2g_lav_dxCrNb() {

   double d2g_lav_dxCrNb_result;
   d2g_lav_dxCrNb_result = 23980778596.6151;
   return d2g_lav_dxCrNb_result;

}

double d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = 23980778596.6151;
   return d2g_lav_dxNbCr_result;

}

double d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 195462020598.431;
   return d2g_lav_dxNbNb_result;

}
