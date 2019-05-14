/******************************************************************************
 *                       Code generated with sympy 1.4                        *
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
   hprime_result = 30.0*pow(x, 2)*pow(1.0 - x, 2);
   return hprime_result;

}

double interface_profile(double z) {

   double interface_profile_result;
   interface_profile_result = 1.0/2.0 - 1.0/2.0*tanh(z);
   return interface_profile_result;

}

double kT() {

   double kT_result;
   kT_result = 1.5782883556379999e-20;
   return kT_result;

}

double RT() {

   double RT_result;
   RT_result = 9504.6886668000006;
   return RT_result;

}

double Vm() {

   double Vm_result;
   Vm_result = 1.0000000000000001e-5;
   return Vm_result;

}

double xe_gam_Cr() {

   double xe_gam_Cr_result;
   xe_gam_Cr_result = 0.52500000000000002;
   return xe_gam_Cr_result;

}

double xe_gam_Nb() {

   double xe_gam_Nb_result;
   xe_gam_Nb_result = 0.017999999999999999;
   return xe_gam_Nb_result;

}

double xe_del_Cr() {

   double xe_del_Cr_result;
   xe_del_Cr_result = 0.0258;
   return xe_del_Cr_result;

}

double xe_del_Nb() {

   double xe_del_Nb_result;
   xe_del_Nb_result = 0.24399999999999999;
   return xe_del_Nb_result;

}

double xe_lav_Cr() {

   double xe_lav_Cr_result;
   xe_lav_Cr_result = 0.375;
   return xe_lav_Cr_result;

}

double xe_lav_Nb() {

   double xe_lav_Nb_result;
   xe_lav_Nb_result = 0.25900000000000001;
   return xe_lav_Nb_result;

}

double matrix_min_Cr() {

   double matrix_min_Cr_result;
   matrix_min_Cr_result = 0.27939999999999998;
   return matrix_min_Cr_result;

}

double matrix_max_Cr() {

   double matrix_max_Cr_result;
   matrix_max_Cr_result = 0.32879999999999998;
   return matrix_max_Cr_result;

}

double matrix_min_Nb() {

   double matrix_min_Nb_result;
   matrix_min_Nb_result = 0.020199999999999999;
   return matrix_min_Nb_result;

}

double matrix_max_Nb() {

   double matrix_max_Nb_result;
   matrix_max_Nb_result = 0.0269;
   return matrix_max_Nb_result;

}

double enrich_min_Cr() {

   double enrich_min_Cr_result;
   enrich_min_Cr_result = 0.24729999999999999;
   return enrich_min_Cr_result;

}

double enrich_max_Cr() {

   double enrich_max_Cr_result;
   enrich_max_Cr_result = 0.29670000000000002;
   return enrich_max_Cr_result;

}

double enrich_min_Nb() {

   double enrich_min_Nb_result;
   enrich_min_Nb_result = 0.16589999999999999;
   return enrich_min_Nb_result;

}

double enrich_max_Nb() {

   double enrich_max_Nb_result;
   enrich_max_Nb_result = 0.1726;
   return enrich_max_Nb_result;

}

double xr_gam_Cr(double P_del, double P_lav) {

   double xr_gam_Cr_result;
   xr_gam_Cr_result = -1.2678788831172791e-9*P_del + 8.3903379061418654e-10*P_lav + 0.52500000000000002;
   return xr_gam_Cr_result;

}

double xr_gam_Nb(double P_del, double P_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 1.914953946717973e-10*P_del - 7.5073446491249387e-11*P_lav + 0.017999999999999999;
   return xr_gam_Nb_result;

}

double xr_del_Cr(double P_del, double P_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = -7.1764784309488426e-11*P_del + 5.6053532605840619e-11*P_lav + 0.0258;
   return xr_del_Cr_result;

}

double xr_del_Nb(double P_del, double P_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = -3.0098560476695523e-12*P_del + 2.8083066899508087e-11*P_lav + 0.24399999999999999;
   return xr_del_Nb_result;

}

double xr_lav_Cr(double P_del, double P_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = -1.7383901583177733e-10*P_del + 1.2866636929017012e-10*P_lav + 0.375;
   return xr_lav_Cr_result;

}

double xr_lav_Nb(double P_del, double P_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = 1.2446553532108367e-11*P_del + 1.3771353705435363e-11*P_lav + 0.25900000000000001;
   return xr_lav_Nb_result;

}

double inv_fict_det(double f_del, double f_gam, double f_lav) {

   double inv_fict_det_result;
   inv_fict_det_result = 1.1017502786047317/(0.028198882496280081*pow(f_del, 2) + 0.58678637752223428*f_del*f_gam + 0.094379501815769914*f_del*f_lav + 1.0*pow(f_gam, 2) + 0.60304745699564377*f_gam*f_lav + 0.061011302923507621*pow(f_lav, 2));
   return inv_fict_det_result;

}

double fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = 0.90764669582512891*INV_DET*(0.50514638806645196*XCR*f_del + 1.0*XCR*f_gam + 0.42609155984759101*XCR*f_lav - 0.16577335655420122*XNB*f_del - 0.26381641966842839*XNB*f_lav + 0.04222033549765767*pow(f_del, 2) + 0.020044914882261487*f_del*f_gam - 0.043567313569073896*f_del*f_lav - 0.27734945844324121*f_gam*f_lav - 0.059424948213882431*pow(f_lav, 2));
   return fict_gam_Cr_result;

}

double fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = -0.90764669582512891*INV_DET*(0.07866923593857271*XCR*f_del + 0.05453834653134719*XCR*f_lav - 0.081639989455782366*XNB*f_del - 1.0*XNB*f_gam - 0.17695589714805279*XNB*f_lav + 0.017382911255062642*pow(f_del, 2) + 0.19360601614705264*f_del*f_gam + 0.031715112323014986*f_del*f_lav + 0.22269771999378604*f_gam*f_lav + 0.024281493959467298*pow(f_lav, 2));
   return fict_gam_Nb_result;

}

double fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 0.15046363933226173*INV_DET*(0.17010503426140242*XCR*f_del + 0.49247955855372533*XCR*f_gam + 0.15530303676586266*XCR*f_lav + 1.0*XNB*f_gam + 0.047031703250520807*XNB*f_lav - 0.25468710035954006*f_del*f_gam - 0.064583265651030716*f_del*f_lav - 0.12091759073302955*pow(f_gam, 2) - 0.43220570445062184*f_gam*f_lav - 0.060924405979911687*pow(f_lav, 2));
   return fict_del_Cr_result;

}

double fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = 0.45849445003651351*INV_DET*(0.15573552102331131*XCR*f_gam + 0.01181924454515282*XCR*f_lav + 0.055823189401030648*XNB*f_del + 1.0*XNB*f_gam + 0.13587030957648552*XNB*f_lav + 0.034411631292859853*f_del*f_gam - 0.0023275279854494272*f_del*f_lav + 0.3832671493269863*pow(f_gam, 2) - 0.03476260265836869*f_gam*f_lav - 0.010152441083121579*pow(f_lav, 2));
   return fict_del_Nb_result;

}

double fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 0.25173531954489675*INV_DET*(0.24746540524461916*XCR*f_del + 0.63802503217891204*XCR*f_gam + 0.2199798884265477*XCR*f_lav - 0.028111117851328069*XNB*f_del + 0.95120582224759476*XNB*f_gam + 0.038601787017348746*pow(f_del, 2) + 0.4154163652037744*f_del*f_gam + 0.036414865679024487*f_del*f_lav + 1.0*pow(f_gam, 2) + 0.21426019198823704*f_gam*f_lav);
   return fict_lav_Cr_result;

}

double fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = -0.38674059641464115*INV_DET*(0.01401212615850367*XCR*f_del - 0.12799677738478332*XCR*f_gam - 0.060421534040585929*XNB*f_del - 1.0*XNB*f_gam - 0.14318824560930238*XNB*f_lav - 0.0027593655114734393*pow(f_del, 2) - 0.11564489921674229*f_del*f_gam - 0.01203607258738176*f_del*f_lav - 0.52265226768031503*pow(f_gam, 2) - 0.056986564033684621*f_gam*f_lav);
   return fict_lav_Nb_result;

}

double s_delta() {

   double s_delta_result;
   s_delta_result = 0.222;
   return s_delta_result;

}

double s_laves() {

   double s_laves_result;
   s_laves_result = 0.27750000000000002;
   return s_laves_result;

}

double g_gam(double XCR, double XNB) {

   double g_gam_result;
   g_gam_result = 2225414588.9007587*pow(XCR - 0.52500000000000002, 2) + (14903715603.987373*XCR - 7824450692.0933714)*(XNB - 0.017999999999999999) + 44805620198.983856*pow(XNB - 0.017999999999999999, 2);
   return g_gam_result;

}

double g_del(double XCR, double XNB) {

   double g_del_result;
   g_del_result = 19076223392.853703*pow(XCR - 0.0258, 2) + (16983307497.06665*XCR - 438169333.42431957)*(XNB - 0.24399999999999999) + 85911591727.909775*pow(XNB - 0.24399999999999999, 2);
   return g_del_result;

}

double g_lav(double XCR, double XNB) {

   double g_lav_result;
   g_lav_result = 8880622842.3634224*pow(XCR - 0.375, 2) + (23980778596.615067*XCR - 8992791973.7306499)*(XNB - 0.25900000000000001) + 97731010299.215485*pow(XNB - 0.25900000000000001, 2);
   return g_lav_result;

}

double dg_gam_dxCr(double XCR, double XNB) {

   double dg_gam_dxCr_result;
   dg_gam_dxCr_result = 4450829177.8015175*XCR + 14903715603.987373*XNB - 2604952199.2175694;
   return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 14903715603.987373*XCR + 89611240397.967712*XNB - 9437453019.2567902;
   return dg_gam_dxNb_result;

}

double dg_del_dxCr(double XCR, double XNB) {

   double dg_del_dxCr_result;
   dg_del_dxCr_result = 38152446785.707405*XCR + 16983307497.06665*XNB - 5128260156.3555136;
   return dg_del_dxCr_result;

}

double dg_del_dxNb(double XCR, double XNB) {

   double dg_del_dxNb_result;
   dg_del_dxNb_result = 16983307497.06665*XCR + 171823183455.81955*XNB - 42363026096.644287;
   return dg_del_dxNb_result;

}

double dg_lav_dxCr(double XCR, double XNB) {

   double dg_lav_dxCr_result;
   dg_lav_dxCr_result = 17761245684.726845*XCR + 23980778596.615067*XNB - 12871488788.295868;
   return dg_lav_dxCr_result;

}

double dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = 23980778596.615067*XCR + 195462020598.43097*XNB - 59617455308.724274;
   return dg_lav_dxNb_result;

}

double d2g_gam_dxCrCr() {

   double d2g_gam_dxCrCr_result;
   d2g_gam_dxCrCr_result = 4450829177.8015175;
   return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 14903715603.987373;
   return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 14903715603.987373;
   return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 89611240397.967712;
   return d2g_gam_dxNbNb_result;

}

double d2g_del_dxCrCr() {

   double d2g_del_dxCrCr_result;
   d2g_del_dxCrCr_result = 38152446785.707405;
   return d2g_del_dxCrCr_result;

}

double d2g_del_dxCrNb() {

   double d2g_del_dxCrNb_result;
   d2g_del_dxCrNb_result = 16983307497.06665;
   return d2g_del_dxCrNb_result;

}

double d2g_del_dxNbCr() {

   double d2g_del_dxNbCr_result;
   d2g_del_dxNbCr_result = 16983307497.06665;
   return d2g_del_dxNbCr_result;

}

double d2g_del_dxNbNb() {

   double d2g_del_dxNbNb_result;
   d2g_del_dxNbNb_result = 171823183455.81955;
   return d2g_del_dxNbNb_result;

}

double d2g_lav_dxCrCr() {

   double d2g_lav_dxCrCr_result;
   d2g_lav_dxCrCr_result = 17761245684.726845;
   return d2g_lav_dxCrCr_result;

}

double d2g_lav_dxCrNb() {

   double d2g_lav_dxCrNb_result;
   d2g_lav_dxCrNb_result = 23980778596.615067;
   return d2g_lav_dxCrNb_result;

}

double d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = 23980778596.615067;
   return d2g_lav_dxNbCr_result;

}

double d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 195462020598.43097;
   return d2g_lav_dxNbNb_result;

}
