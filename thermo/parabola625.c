/******************************************************************************
 *                      Code generated with sympy 1.5.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/
#include "parabola625.h"
#include <math.h>

double p(double x) {

   double p_result;
   p_result = pow(x, 3)*(6.0*pow(x, 2) - 15.0*x + 10.0);
   return p_result;

}

double pPrime(double x) {

   double pPrime_result;
   pPrime_result = 30.0*pow(x, 2)*pow(1.0 - x, 2);
   return pPrime_result;

}

double interface_profile(double z) {

   double interface_profile_result;
   interface_profile_result = 1.0/2.0 - 1.0/2.0*tanh(z);
   return interface_profile_result;

}

double kT() {

   double kT_result;
   kT_result = 1.5782889043500002e-20;
   return kT_result;

}

double RT() {

   double RT_result;
   RT_result = 9504.6840941999999;
   return RT_result;

}

double Vm() {

   double Vm_result;
   Vm_result = 1.0000000000000001e-5;
   return Vm_result;

}

double xe_gam_Cr() {

   double xe_gam_Cr_result;
   xe_gam_Cr_result = 0.52421634830562147;
   return xe_gam_Cr_result;

}

double xe_gam_Nb() {

   double xe_gam_Nb_result;
   xe_gam_Nb_result = 0.01299272922003303;
   return xe_gam_Nb_result;

}

double xe_del_Cr() {

   double xe_del_Cr_result;
   xe_del_Cr_result = 0.022966218927631978;
   return xe_del_Cr_result;

}

double xe_del_Nb() {

   double xe_del_Nb_result;
   xe_del_Nb_result = 0.24984563695705883;
   return xe_del_Nb_result;

}

double xe_lav_Cr() {

   double xe_lav_Cr_result;
   xe_lav_Cr_result = 0.37392129441013022;
   return xe_lav_Cr_result;

}

double xe_lav_Nb() {

   double xe_lav_Nb_result;
   xe_lav_Nb_result = 0.25826261799015571;
   return xe_lav_Nb_result;

}

double matrix_min_Cr() {

   double matrix_min_Cr_result;
   matrix_min_Cr_result = 0.30851614493185742;
   return matrix_min_Cr_result;

}

double matrix_max_Cr() {

   double matrix_max_Cr_result;
   matrix_max_Cr_result = 0.36263227633734074;
   return matrix_max_Cr_result;

}

double matrix_min_Nb() {

   double matrix_min_Nb_result;
   matrix_min_Nb_result = 0.019424801579942592;
   return matrix_min_Nb_result;

}

double matrix_max_Nb() {

   double matrix_max_Nb_result;
   matrix_max_Nb_result = 0.025522709998118311;
   return matrix_max_Nb_result;

}

double enrich_min_Cr() {

   double enrich_min_Cr_result;
   enrich_min_Cr_result = 0.29783560803725534;
   return enrich_min_Cr_result;

}

double enrich_max_Cr() {

   double enrich_max_Cr_result;
   enrich_max_Cr_result = 0.35636564894177969;
   return enrich_max_Cr_result;

}

double enrich_min_Nb() {

   double enrich_min_Nb_result;
   enrich_min_Nb_result = 0.15335241484365611;
   return enrich_min_Nb_result;

}

double enrich_max_Nb() {

   double enrich_max_Nb_result;
   enrich_max_Nb_result = 0.15955557903581488;
   return enrich_max_Nb_result;

}

double xr_gam_Cr(double r_del, double r_lav) {

   double xr_gam_Cr_result;
   xr_gam_Cr_result = 0.52421634830562147 - 4.7768723157796301e-5/r_lav + 5.3413073820986343e-5/r_del;
   return xr_gam_Cr_result;

}

double xr_gam_Nb(double r_del, double r_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 0.01299272922003303 + 3.851373946831059e-6/r_lav - 3.0197595896037847e-6/r_del;
   return xr_gam_Nb_result;

}

double xr_del_Cr(double r_del, double r_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = 0.022966218927631978 + 1.6436514881430556e-11/r_lav - 1.7091137328821305e-11/r_del;
   return xr_del_Cr_result;

}

double xr_del_Nb(double r_del, double r_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = 0.24984563695705883 + 1.9676894087978162e-13/r_lav - 2.5675808729451331e-14/r_del;
   return xr_del_Nb_result;

}

double xr_lav_Cr(double r_del, double r_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = 0.37392129441013022 + 3.5336079342738321e-11/r_lav - 4.574562112642643e-11/r_del;
   return xr_lav_Cr_result;

}

double xr_lav_Nb(double r_del, double r_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = 0.25826261799015571 + 3.2416844758802665e-12/r_lav + 3.3534115774403428e-12/r_del;
   return xr_lav_Nb_result;

}

double inv_fict_det(double pDel, double pGam, double pLav) {

   double inv_fict_det_result;
   inv_fict_det_result = 2.0958068891718297e-12/(4.7849153889199254e-14*pow(pDel, 2) + 1.1236727699162962e-7*pDel*pGam + 1.9298640325127681e-12*pDel*pLav - 1.0*pow(pGam, 2) - 4.2929348472764003e-6*pGam*pLav + 4.3406019534064548e-12*pow(pLav, 2));
   return inv_fict_det_result;

}

double fict_gam_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = -477143197289.11456*INV_DET*(1.5104100862826891e-7*XCR*pDel + 1.0*XCR*pGam + 5.423310547250087e-6*XCR*pLav - 1.000639631826674e-6*XNB*pDel + 4.8451604002149525e-6*XNB*pLav + 2.4653658022369671e-7*pow(pDel, 2) - 0.022966344009521668*pDel*pGam - 1.1331457738328127e-6*pDel*pLav - 0.37392194992340644*pGam*pLav - 3.2792173847721e-6*pow(pLav, 2));
   return fict_gam_Cr_result;

}

double fict_gam_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = 477143197289.11475*INV_DET*(8.0585462454713149e-9*XCR*pDel + 3.6939468557875602e-7*XCR*pLav + 2.634082856198985e-7*XNB*pDel - 1.0*XNB*pGam + 1.1303756999736869e-6*XNB*pLav - 6.5996484616091373e-8*pow(pDel, 2) + 0.24984563077020269*pDel*pGam - 3.6194478639387638e-7*pDel*pLav + 0.25826235388381708*pGam*pLav - 4.3005827017122384e-7*pow(pLav, 2));
   return fict_gam_Nb_result;

}

double fict_del_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 10958234810.744873*INV_DET*(2.0834467109506573e-12*XCR*pDel + 1.1469317254443785e-5*XCR*pGam + 7.82961317250236e-11*XCR*pLav - 4.356982684800926e-5*XNB*pGam + 1.048209552997839e-10*XNB*pLav + 1.0734689862761115e-5*pDel*pGam - 2.6836416014828854e-11*pDel*pLav - 1.0*pow(pGam, 2) + 2.670881720729014e-6*pGam*pLav - 5.2007346944746807e-11*pow(pLav, 2));
   return fict_del_Cr_result;

}

double fict_del_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = -119212143094.41014*INV_DET*(3.2254101144891419e-8*XCR*pGam - 4.8388867181311764e-14*XCR*pLav - 1.9151487156967273e-13*XNB*pDel + 6.0453732235641904e-7*XNB*pGam - 5.270780053071001e-13*XNB*pLav - 2.6414904440251002e-7*pDel*pGam - 1.7476034989064573e-12*pDel*pLav + 1.0*pow(pGam, 2) + 4.1247450990008859e-6*pGam*pLav - 4.186383887504434e-12*pow(pLav, 2));
   return fict_del_Nb_result;

}

double fict_lav_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 178414314723.03433*INV_DET*(3.521808140121955e-13*XCR*pDel + 3.023025795103046e-6*XCR*pGam + 1.1608310114705962e-11*XCR*pLav - 6.4381192901744587e-12*XNB*pDel + 1.295767847062048e-5*XNB*pGam + 1.6482968231884889e-12*pow(pDel, 2) - 3.1944799241780134e-6*pDel*pGam + 3.1942992948134332e-12*pDel*pLav - 1.0*pow(pGam, 2) - 8.7697910899421908e-6*pGam*pLav);
   return fict_lav_Cr_result;

}

double fict_lav_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = -123228125271.53731*INV_DET*(4.6811882805842998e-14*XCR*pDel + 1.4303079021146584e-6*XCR*pGam - 6.9625939234668684e-12*XNB*pDel + 2.0999229913663052e-5*XNB*pGam - 1.6806948005124795e-11*XNB*pLav + 1.6906494189119086e-12*pow(pDel, 2) - 5.391782063778788e-6*pDel*pGam + 4.0499503984630051e-12*pDel*pLav + 1.0*pow(pGam, 2) - 1.6651992197232569e-6*pGam*pLav);
   return fict_lav_Nb_result;

}

double s_delta() {

   double s_delta_result;
   s_delta_result = 0.13;
   return s_delta_result;

}

double s_laves() {

   double s_laves_result;
   s_laves_result = 0.13;
   return s_laves_result;

}

double CALPHAD_gam(double XCR, double XNB) {

   double CALPHAD_gam_result;
   CALPHAD_gam_result = -2111099999.9999998*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*pow(-XCR - XNB + 1, 2) + 1474821796.9999995*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 669388631.5*XCR*(-XCR - XNB + 1) + 1707300680.1792686*XCR + 6973684804.4499989*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 7846288044.7499981*XNB*(-XCR - XNB + 1) + 1069046462.7507229*XNB + 950472067.50000012*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 950472067.50000012*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 950472067.50000012*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 950472067.50000012*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -0.00075260355561858024*(-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -1.0038219457461011e-80*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 25) - 1.8216695466490673e-49*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15) - 2.186816584109128e-17*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 5464277694.8364201;
   return CALPHAD_gam_result;

}

double CALPHAD_del(double XCR, double XNB) {

   double CALPHAD_del_result;
   CALPHAD_del_result = -21668797081.409531*XCR*XNB - 5258769591.26929*XCR*(1 - 4*XNB) - 1231849999.9999998*XNB*(1 - 1.3333333333333333*XCR)*(1 - 4*XNB) - 34242601639.394947*XNB*(1 - 1.3333333333333333*XCR) - 4964277694.8364201*(1 - 1.3333333333333333*XCR)*(1 - 4*XNB) + 712854050.625*((1.3333333333333333*XCR > 1.0e-13) ? (
      1.3333333333333333*XCR*log(1.3333333333333333*XCR)
   )
   : (
      0
   )) + 237618016.87500003*((4*XNB > 1.0e-13) ? (
      4*XNB*log(4*XNB)
   )
   : (
      0
   )) + 712854050.625*((1.3333333333333333*XCR - 1 < -1.0e-13) ? (
      (1 - 1.3333333333333333*XCR)*log(1 - 1.3333333333333333*XCR)
   )
   : (
      0
   )) + 237618016.87500003*((4*XNB - 1 < -1.0e-13) ? (
      (1 - 4*XNB)*log(1 - 4*XNB)
   )
   : (
      0
   ));
   return CALPHAD_del_result;

}

double CALPHAD_lav(double XCR, double XNB) {

   double CALPHAD_lav_result;
   CALPHAD_lav_result = -46695351257.249992*XNB*(1 - 3*XNB)*(-1.5*XCR - 1.5*XNB + 1.5) + 1851135999.9999998*XNB*(1 - 3*XNB)*(1.5*XCR + 1.5*XNB - 0.5) - 10298680536.599998*XNB*(-1.5*XCR - 1.5*XNB + 1.5)*(1.5*XCR + 1.5*XNB - 0.5) - 22164936866.458534*XNB*(-1.5*XCR - 1.5*XNB + 1.5) - 16694022288.40456*XNB*(1.5*XCR + 1.5*XNB - 0.5) + 4855811416.8900013*(1 - 3*XNB)*(-1.5*XCR - 1.5*XNB + 1.5) - 4004010359.6571507*(1 - 3*XNB)*(1.5*XCR + 1.5*XNB - 0.5) + 316824022.5*((3*XNB > 1.0e-13) ? (
      3*XNB*log(3*XNB)
   )
   : (
      0
   )) + 316824022.5*((3*XNB - 1 < -1.0e-13) ? (
      (1 - 3*XNB)*log(1 - 3*XNB)
   )
   : (
      0
   )) + 633648045.0*((1.5*XCR + 1.5*XNB - 1.5 < -1.0e-13) ? (
      (-1.5*XCR - 1.5*XNB + 1.5)*log(-1.5*XCR - 1.5*XNB + 1.5)
   )
   : (
      0
   )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 1.0e-13) ? (
      (1.5*XCR + 1.5*XNB - 0.5)*log(1.5*XCR + 1.5*XNB - 0.5)
   )
   : (
      0
   ));
   return CALPHAD_lav_result;

}

double g_gam(double XCR, double XNB) {

   double g_gam_result;
   g_gam_result = -5554.7222271853516*pow(XCR - 0.52421634830562147, 2) + (45273.868191803347*XCR - 23733.301857177179)*(XNB - 0.01299272922003303) + 474475.92148860614*pow(XNB - 0.01299272922003303, 2);
   return g_gam_result;

}

double g_del(double XCR, double XNB) {

   double g_del_result;
   g_del_result = 21346492990.798885*pow(XCR - 0.022966218927631978, 2) + (16906497386.287418*XCR - 388278320.27291465)*(XNB - 0.24984563695705883) + 3085369132931.8848*pow(XNB - 0.24984563695705883, 2);
   return g_del_result;

}

double g_lav(double XCR, double XNB) {

   double g_lav_result;
   g_lav_result = 8866730284.8069954*pow(XCR - 0.37392129441013022, 2) + (24191004361.532181*XCR - 9045531663.945219)*(XNB - 0.25826261799015571) + 98294310279.883911*pow(XNB - 0.25826261799015571, 2);
   return g_lav_result;

}

double dg_gam_dxCr(double XCR, double XNB) {

   double dg_gam_dxCr_result;
   dg_gam_dxCr_result = -11109.444454370703*XCR + 45273.868191803347*XNB + 5235.5212934147803;
   return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 45273.868191803347*XCR + 948951.84297721228*XNB - 36062.7761958314;
   return dg_gam_dxNb_result;

}

double dg_del_dxCr(double XCR, double XNB) {

   double dg_del_dxCr_result;
   dg_del_dxCr_result = 42692985981.597771*XCR + 16906497386.287418*XNB - 5204511070.9175282;
   return dg_del_dxCr_result;

}

double dg_del_dxNb(double XCR, double XNB) {

   double dg_del_dxNb_result;
   dg_del_dxNb_result = 16906497386.287418*XCR + 6170738265863.7695*XNB - 1542120310850.303;
   return dg_del_dxNb_result;

}

double dg_lav_dxCr(double XCR, double XNB) {

   double dg_lav_dxCr_result;
   dg_lav_dxCr_result = 17733460569.613991*XCR + 24191004361.532181*XNB - 12878550648.781645;
   return dg_lav_dxCr_result;

}

double dg_lav_dxNb(double XCR, double XNB) {

   double dg_lav_dxNb_result;
   dg_lav_dxNb_result = 24191004361.532181*XCR + 196588620559.76782*XNB - 59817023476.78421;
   return dg_lav_dxNb_result;

}

double d2g_gam_dxCrCr() {

   double d2g_gam_dxCrCr_result;
   d2g_gam_dxCrCr_result = -11109.444454370703;
   return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 45273.868191803347;
   return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 45273.868191803347;
   return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 948951.84297721228;
   return d2g_gam_dxNbNb_result;

}

double d2g_del_dxCrCr() {

   double d2g_del_dxCrCr_result;
   d2g_del_dxCrCr_result = 42692985981.597771;
   return d2g_del_dxCrCr_result;

}

double d2g_del_dxCrNb() {

   double d2g_del_dxCrNb_result;
   d2g_del_dxCrNb_result = 16906497386.287418;
   return d2g_del_dxCrNb_result;

}

double d2g_del_dxNbCr() {

   double d2g_del_dxNbCr_result;
   d2g_del_dxNbCr_result = 16906497386.287418;
   return d2g_del_dxNbCr_result;

}

double d2g_del_dxNbNb() {

   double d2g_del_dxNbNb_result;
   d2g_del_dxNbNb_result = 6170738265863.7695;
   return d2g_del_dxNbNb_result;

}

double d2g_lav_dxCrCr() {

   double d2g_lav_dxCrCr_result;
   d2g_lav_dxCrCr_result = 17733460569.613991;
   return d2g_lav_dxCrCr_result;

}

double d2g_lav_dxCrNb() {

   double d2g_lav_dxCrNb_result;
   d2g_lav_dxCrNb_result = 24191004361.532181;
   return d2g_lav_dxCrNb_result;

}

double d2g_lav_dxNbCr() {

   double d2g_lav_dxNbCr_result;
   d2g_lav_dxNbCr_result = 24191004361.532181;
   return d2g_lav_dxNbCr_result;

}

double d2g_lav_dxNbNb() {

   double d2g_lav_dxNbNb_result;
   d2g_lav_dxNbNb_result = 196588620559.76782;
   return d2g_lav_dxNbNb_result;

}

double L0_Cr(double XCR, double XNB) {

   double L0_Cr_result;
   L0_Cr_result = 3.5027570743586952e-21*XCR*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR);
   return L0_Cr_result;

}

double L0_Nb(double XCR, double XNB) {

   double L0_Nb_result;
   L0_Nb_result = 1.4170990277916469e-20*XNB*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB);
   return L0_Nb_result;

}

double L0_Ni(double XCR, double XNB) {

   double L0_Ni_result;
   L0_Ni_result = 1.8295676400933012e-21*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return L0_Ni_result;

}

double D_CrCr(double XCR, double XNB) {

   double D_CrCr_result;
   D_CrCr_result = -1.4170990277916469e-20*XCR*XNB*(55257.013603585379*XCR + 42491.92925501446*XNB - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 48874.471429299927 - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.4170990277916469e-20*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB - 21110.999999999996*pow(XCR, 2) + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 7036.9999999999991*XCR*(3*XCR + 3*XNB - 3) + 180313.59600758535*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 378791.20188051439*XNB - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 228627.04116400718)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 3.5027570743586952e-21*XCR*(1 - XCR)*(4649.5776635853872*XCR - 8115.506684985532*XNB - 1.9999999999999998*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 9504.7206750000023*((XCR > 1.0e-13) ? (
      1.0/XCR
   )
   : (
      0
   )) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (3127984108.0821762 + 2.3268190314398873e+27/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 6) + 3.1155389261560675e+45/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 12))*pow(-3605*XCR - 3605*XNB + 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*pow(-3605*XCR - 3605*XNB + 4714, 2)*(6.0229316744766066e-78*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 3.8255060479630414e-47*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 4.3736331682182561e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 4649.5776635853836 + 19009.441350000005*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 19009.441350000005*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 181510.70025840754*pow(-0.43707093821510296*XCR - 0.43707093821510296*XNB + 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 3.5027570743586952e-21*XCR*(1 - XCR)*(21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 99832.771798585367*XCR - 21110.999999999996*pow(XNB, 2) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 7036.9999999999991*XNB*(3*XCR + 3*XNB - 3) + 129340.24564251443*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      -3127984108.0821762*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17) + 2.7131358180049818*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2819095677409.0615*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 3.8255060479630414e-47*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 4.3736331682182561e-16*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 21)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 11)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 53263.329056792689 + 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*(42221.999999999993*pow(XCR, 2)*XNB + 42221.999999999993*XCR*pow(XNB, 2) - 14073.999999999998*XCR*XNB*(3*XCR + 3*XNB - 3) + 29496.435939999992*XCR*(-XCR - XNB + 1) - 29496.435939999992*XCR*(2*XCR + XNB - 1) + 18037.35029358539*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1) - 139473.69608899998*XNB*(XCR + 2*XNB - 1) + 38832.99406101444*XNB - 34674.171494467504*pow(XCR - 0.27225130890052357, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) + 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      pow(3605*XCR - 633, 2)*(3127984108.0821762 + 2.3268190314398873e+27/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 6) + 3.1155389261560675e+45/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 12))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)
   )
   : (
      pow(3605*XCR - 633, 2)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 3.8255060479630414e-47*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 4.3736331682182561e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      1.0/(-XCR - XNB + 1)
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) - 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 99832.771798585367*XCR - 21110.999999999996*pow(XNB, 2) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 7036.9999999999991*XNB*(3*XCR + 3*XNB - 3) + 129340.24564251443*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      -3127984108.0821762*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17) + 2.7131358180049818*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2819095677409.0615*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 3.8255060479630414e-47*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 4.3736331682182561e-16*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 21)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 11)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 53263.329056792689 + 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_CrCr_result;

}

double D_CrNb(double XCR, double XNB) {

   double D_CrNb_result;
   D_CrNb_result = -1.4170990277916469e-20*XCR*XNB*(-105327.6824854146*XCR - 118092.76683398552*XNB - 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 9504.7206750000023*((XNB > 1.0e-13) ? (
      1.0/XNB
   )
   : (
      0
   )) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 118092.7668339855)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.4170990277916469e-20*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB - 21110.999999999996*pow(XCR, 2) + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 7036.9999999999991*XCR*(3*XCR + 3*XNB - 3) + 180313.59600758535*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 378791.20188051439*XNB - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 228627.04116400718)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 3.5027570743586952e-21*XCR*(1 - XCR)*(55257.013603585379*XCR + 42491.92925501446*XNB - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 48874.471429299927 - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 3.5027570743586952e-21*XCR*(1 - XCR)*(21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 99832.771798585367*XCR - 21110.999999999996*pow(XNB, 2) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 7036.9999999999991*XNB*(3*XCR + 3*XNB - 3) + 129340.24564251443*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      -3127984108.0821762*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17) + 2.7131358180049818*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2819095677409.0615*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 3.8255060479630414e-47*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 4.3736331682182561e-16*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 21)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 11)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 53263.329056792689 + 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*(42221.999999999993*pow(XCR, 2)*XNB + 42221.999999999993*XCR*pow(XNB, 2) - 14073.999999999998*XCR*XNB*(3*XCR + 3*XNB - 3) + 29496.435939999992*XCR*(-XCR - XNB + 1) - 29496.435939999992*XCR*(2*XCR + XNB - 1) + 18037.35029358539*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1) - 139473.69608899998*XNB*(XCR + 2*XNB - 1) + 38832.99406101444*XNB - 34674.171494467504*pow(XCR - 0.27225130890052357, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) + 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      pow(3605*XCR - 633, 2)*(3127984108.0821762 + 2.3268190314398873e+27/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 6) + 3.1155389261560675e+45/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 12))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)
   )
   : (
      pow(3605*XCR - 633, 2)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 3.8255060479630414e-47*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 4.3736331682182561e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      1.0/(-XCR - XNB + 1)
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) - 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB - 21110.999999999996*pow(XCR, 2) + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 7036.9999999999991*XCR*(3*XCR + 3*XNB - 3) + 180313.59600758535*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 378791.20188051439*XNB - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 228627.04116400718)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_CrNb_result;

}

double D_NbCr(double XCR, double XNB) {

   double D_NbCr_result;
   D_NbCr_result = -3.5027570743586952e-21*XCR*XNB*(4649.5776635853872*XCR - 8115.506684985532*XNB - 1.9999999999999998*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 9504.7206750000023*((XCR > 1.0e-13) ? (
      1.0/XCR
   )
   : (
      0
   )) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (3127984108.0821762 + 2.3268190314398873e+27/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 6) + 3.1155389261560675e+45/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 12))*pow(-3605*XCR - 3605*XNB + 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*pow(-3605*XCR - 3605*XNB + 4714, 2)*(6.0229316744766066e-78*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 3.8255060479630414e-47*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 4.3736331682182561e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 4649.5776635853836 + 19009.441350000005*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 19009.441350000005*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 181510.70025840754*pow(-0.43707093821510296*XCR - 0.43707093821510296*XNB + 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 3.5027570743586952e-21*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 99832.771798585367*XCR - 21110.999999999996*pow(XNB, 2) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 7036.9999999999991*XNB*(3*XCR + 3*XNB - 3) + 129340.24564251443*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      -3127984108.0821762*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17) + 2.7131358180049818*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2819095677409.0615*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 3.8255060479630414e-47*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 4.3736331682182561e-16*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 21)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 11)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 53263.329056792689 + 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.4170990277916469e-20*XNB*(1 - XNB)*(55257.013603585379*XCR + 42491.92925501446*XNB - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 48874.471429299927 - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.4170990277916469e-20*XNB*(1 - XNB)*(21110.999999999996*pow(XCR, 2)*XNB - 21110.999999999996*pow(XCR, 2) + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 7036.9999999999991*XCR*(3*XCR + 3*XNB - 3) + 180313.59600758535*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 378791.20188051439*XNB - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 228627.04116400718)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.8295676400933012e-21*XNB*(-XCR - XNB + 1)*(42221.999999999993*pow(XCR, 2)*XNB + 42221.999999999993*XCR*pow(XNB, 2) - 14073.999999999998*XCR*XNB*(3*XCR + 3*XNB - 3) + 29496.435939999992*XCR*(-XCR - XNB + 1) - 29496.435939999992*XCR*(2*XCR + XNB - 1) + 18037.35029358539*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1) - 139473.69608899998*XNB*(XCR + 2*XNB - 1) + 38832.99406101444*XNB - 34674.171494467504*pow(XCR - 0.27225130890052357, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) + 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      pow(3605*XCR - 633, 2)*(3127984108.0821762 + 2.3268190314398873e+27/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 6) + 3.1155389261560675e+45/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 12))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)
   )
   : (
      pow(3605*XCR - 633, 2)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 3.8255060479630414e-47*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 4.3736331682182561e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      1.0/(-XCR - XNB + 1)
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) - 1.8295676400933012e-21*XNB*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 99832.771798585367*XCR - 21110.999999999996*pow(XNB, 2) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 7036.9999999999991*XNB*(3*XCR + 3*XNB - 3) + 129340.24564251443*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      -3127984108.0821762*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17) + 2.7131358180049818*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2819095677409.0615*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 3.8255060479630414e-47*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 4.3736331682182561e-16*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 21)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 11)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 53263.329056792689 + 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_NbCr_result;

}

double D_NbNb(double XCR, double XNB) {

   double D_NbNb_result;
   D_NbNb_result = -3.5027570743586952e-21*XCR*XNB*(55257.013603585379*XCR + 42491.92925501446*XNB - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 48874.471429299927 - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 3.5027570743586952e-21*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 99832.771798585367*XCR - 21110.999999999996*pow(XNB, 2) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 7036.9999999999991*XNB*(3*XCR + 3*XNB - 3) + 129340.24564251443*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*XCR*XNB + 44244.653909999994*XCR - 21111*pow(XNB, 2) + 7037*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XNB - 21442.104285000001) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875 - (-2819095677409.0615*XCR - 2819095677409.0615*XNB + 3686329271374.8447)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (-8.3881826083407944e+29*XCR - 8.3881826083407944e+29*XNB + 1.0968624914207629e+30)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (-7.0196986429953896e+47*XCR - 7.0196986429953896e+47*XNB + 9.1791565611873145e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(-3605*XCR - 3605*XNB + 4714)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      -3127984108.0821762*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17) + 2.7131358180049818*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2819095677409.0615*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 3.8255060479630414e-47*(3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 4.3736331682182561e-16*(3605*XCR - 633)*(-3605*XCR - 3605*XNB + 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 21)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 11)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 53263.329056792689 + 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 9504.7206750000023*(-1.9099999999999999*XCR - 1.9099999999999999*XNB + 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.4170990277916469e-20*XNB*(1 - XNB)*(-105327.6824854146*XCR - 118092.76683398552*XNB - 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 9504.7206750000023*((XNB > 1.0e-13) ? (
      1.0/XNB
   )
   : (
      0
   )) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 118092.7668339855)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.4170990277916469e-20*XNB*(1 - XNB)*(21110.999999999996*pow(XCR, 2)*XNB - 21110.999999999996*pow(XCR, 2) + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 7036.9999999999991*XCR*(3*XCR + 3*XNB - 3) + 180313.59600758535*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 378791.20188051439*XNB - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 228627.04116400718)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.8295676400933012e-21*XNB*(-XCR - XNB + 1)*(42221.999999999993*pow(XCR, 2)*XNB + 42221.999999999993*XCR*pow(XNB, 2) - 14073.999999999998*XCR*XNB*(3*XCR + 3*XNB - 3) + 29496.435939999992*XCR*(-XCR - XNB + 1) - 29496.435939999992*XCR*(2*XCR + XNB - 1) + 18037.35029358539*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1) - 139473.69608899998*XNB*(XCR + 2*XNB - 1) + 38832.99406101444*XNB - 34674.171494467504*pow(XCR - 0.27225130890052357, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/pow((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1, 2) + 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 19009.441350000005*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      pow(3605*XCR - 633, 2)*(3127984108.0821762 + 2.3268190314398873e+27/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 6) + 3.1155389261560675e+45/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 12))*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)
   )
   : (
      pow(3605*XCR - 633, 2)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3)*(6.0229316744766066e-78*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 3.8255060479630414e-47*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 4.3736331682182561e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      1.0/(-XCR - XNB + 1)
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 + 260665342.34018135/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 3) + 2.5853544793776527e+25/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 9) + 1.2981412192316947e+43/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 15)
   )
   : (
      pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) - 1.8295676400933012e-21*XNB*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB - 21110.999999999996*pow(XCR, 2) + 21110.999999999996*XCR*pow(XNB, 2) - 7036.9999999999991*XCR*XNB*(3*XCR + 3*XNB - 3) - 21110.999999999996*XCR*XNB + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 7036.9999999999991*XCR*(3*XCR + 3*XNB - 3) + 180313.59600758535*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 378791.20188051439*XNB - 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))/((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) - 1) - 0.99999999999999989*(-XCR - XNB + 1)*(-21111*pow(XCR, 2) - 21111*XCR*XNB + 7037*XCR*(3*XCR + 3*XNB - 3) + 69736.848044500002*XCR + 209210.54413350002*XNB - 148199.72849199999) + 1.9999999999999998*(-XCR - XNB + 1)*(-21111*pow(XCR, 2)*XNB - 21111*XCR*pow(XNB, 2) + 7037*XCR*XNB*(3*XCR + 3*XNB - 3) + 14748.217969999998*XCR*(2*XCR + XNB - 1) - 6693.8863150000016*XCR + 69736.848044500002*XNB*(XCR + 2*XNB - 1) - 78462.880447499992*XNB) + 19009.441350000005*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) + 19009.441350000005*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) + 19009.441350000005*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      (2.7131358180049818*XCR - (2819095677409.0615*XCR - 495003485104.00439)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4) - (8.3881826083407944e+29*XCR - 1.4728764469014486e+29)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) - (7.0196986429953896e+47*XCR - 1.2325850876604941e+47)/pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 16) - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      (3605*XCR - 633)*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 4)*(2.5095548643652528e-79*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 20) + 2.732504319973601e-48*pow((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.0000000000000001e-9, 10) + 1.093408292054564e-16)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(3605.0*XCR*(-XCR - XNB + 1) + 1742.0*XCR + 633.0*XNB - 633.00000000099999) > -1
   ))) ? (
      0.00075260355561858024*(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 0.99999999999924738 - 260665342.34018135/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3) - 2.5853544793776527e+25/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 9) - 1.2981412192316947e+43/pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 15)
   )
   : (
      -pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5)*(1.0038219457461011e-80*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 20) + 1.8216695466490673e-49*pow(-(3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 2.186816584109128e-17)
   ))*log(-(1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 228627.04116400718)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_NbNb_result;

}
