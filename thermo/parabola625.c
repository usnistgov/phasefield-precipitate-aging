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
   xr_gam_Cr_result = 0.52421634830562147 - 4.7768723157799601e-10/r_lav + 5.341307382098825e-10/r_del;
   return xr_gam_Cr_result;

}

double xr_gam_Nb(double r_del, double r_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 0.01299272922003303 + 3.8513739468312897e-11/r_lav - 3.0197595896039241e-11/r_del;
   return xr_gam_Nb_result;

}

double xr_del_Cr(double r_del, double r_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = 0.022966218927631978 + 1.6436514881431651e-11/r_lav - 1.7091137328821955e-11/r_del;
   return xr_del_Cr_result;

}

double xr_del_Nb(double r_del, double r_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = 0.24984563695705883 + 1.9676894087979051e-13/r_lav - 2.567580872945739e-14/r_del;
   return xr_del_Nb_result;

}

double xr_lav_Cr(double r_del, double r_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = 0.37392129441013022 + 3.5336079342740874e-11/r_lav - 4.5745621126427884e-11/r_del;
   return xr_lav_Cr_result;

}

double xr_lav_Nb(double r_del, double r_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = 0.25826261799015571 + 3.2416844758802984e-12/r_lav + 3.3534115774402923e-12/r_del;
   return xr_lav_Nb_result;

}

double inv_fict_det(double pDel, double pGam, double pLav) {

   double inv_fict_det_result;
   inv_fict_det_result = 0.0052395172229295749/(0.00047849153889199228*pow(pDel, 2) + 0.011236727699162964*pDel*pGam + 0.019298640325127667*pDel*pLav - 1.0*pow(pGam, 2) - 0.42929348472764017*pGam*pLav + 0.043406019534064519*pow(pLav, 2));
   return inv_fict_det_result;

}

double fict_gam_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = -190.85727891564588*INV_DET*(0.01510410086282689*XCR*pDel + 1.0*XCR*pGam + 0.54233105472500875*XCR*pLav - 0.10006396318266712*XNB*pDel + 0.48451604002149579*XNB*pLav + 0.024402827443487329*pow(pDel, 2) - 0.035474407909427158*pDel*pGam - 0.12343113897515585*pDel*pLav - 0.43947262228625189*pGam*pLav - 0.35067565599038975*pow(pLav, 2));
   return fict_gam_Cr_result;

}

double fict_gam_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = 190.85727891564588*INV_DET*(0.00080585462454712989*XCR*pDel + 0.036939468557875541*XCR*pLav + 0.026340828561989854*XNB*pDel - 1.0*XNB*pGam + 0.11303756999736861*XNB*pLav - 0.0065934316127793354*pow(pDel, 2) + 0.24922695129589076*pDel*pGam - 0.035943739138748466*pDel*pLav + 0.23185198413273903*pGam*pLav - 0.0424418699984234*pow(pLav, 2));
   return fict_gam_Nb_result;

}

double fict_del_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 19.097935730559218*INV_DET*(0.0047818567611449111*XCR*pDel + 0.26323990899605459*XCR*pGam + 0.17970264604942116*XCR*pLav - 1.0*XNB*pGam + 0.2405815282797559*XNB*pLav + 0.2438722859591308*pDel*pGam - 0.061594038710427026*pDel*pLav - 0.35451731853423096*pow(pGam, 2) - 0.036026724034010187*pGam*pLav - 0.11936551211500436*pow(pLav, 2));
   return fict_del_Cr_result;

}

double fict_del_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = -47.566777756775913*INV_DET*(0.00323341685302081*XCR*pGam - 0.00048508987412107887*XCR*pLav - 0.0019199028692683828*XNB*pDel + 0.060603802214371207*XNB*pGam - 0.00528386420554926*XNB*pLav - 0.026455532110375126*pDel*pGam - 0.017519417392467818*pDel*pLav + 1.0*pow(pGam, 2) + 0.41382137475490838*pGam*pLav - 0.041967761415095457*pow(pLav, 2));
   return fict_del_Nb_result;

}

double fict_lav_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 92.47341298948686*INV_DET*(0.0027179314165782864*XCR*pDel + 0.23329995430564826*XCR*pGam + 0.089586341728003019*XCR*pLav - 0.049685746600148163*XNB*pDel + 1.0*XNB*pGam + 0.01272061833395341*pow(pDel, 2) - 0.24731103263889553*pDel*pGam + 0.024651786985269038*pDel*pLav - 0.90703420730251672*pow(pGam, 2) - 0.72376480245077512*pGam*pLav);
   return fict_lav_Cr_result;

}

double fict_lav_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = -103.5078293762674*INV_DET*(0.00022292190236645307*XCR*pDel + 0.068112397835314545*XCR*pGam - 0.033156425031265944*XNB*pDel + 1.0*XNB*pGam - 0.080036020721833315*XNB*pLav + 0.0080510067553091081*pow(pDel, 2) - 0.25644701246272772*pDel*pGam + 0.019286185327338654*pDel*pLav + 0.42751006440208478*pow(pGam, 2) - 0.07825823291632035*pGam*pLav);
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
   g_gam_result = -555472222.71853578*pow(XCR - 0.52421634830562147, 2) + (4527386819.1803322*XCR - 2373330185.7177167)*(XNB - 0.01299272922003303) + 47447592148.860611*pow(XNB - 0.01299272922003303, 2);
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
   dg_gam_dxCr_result = -1110944445.4370716*XCR + 4527386819.1803322*XNB + 523552129.34147865;
   return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 4527386819.1803322*XCR + 94895184297.721222*XNB - 3606277619.5831385;
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
   d2g_gam_dxCrCr_result = -1110944445.4370716;
   return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 4527386819.1803322;
   return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 4527386819.1803322;
   return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 94895184297.721222;
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
   D_CrCr_result = -1.4170990277916469e-20*XCR*XNB*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 42884.208569999995*XCR*(-XCR - XNB + 1) - 8738.1949664146214*XCR + 42221.999999999993*pow(XNB, 2)*(-XCR - XNB + 1) + 42221.999999999993*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 21503.279314985535*XNB - 29496.435939999992*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 1.0e-13) ? (
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
   )) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17)
   )
   : (
      -6.0229316744766066e-78*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 8738.1949664146177 + 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 181510.70025840754*pow(0.43707093821510296*XCR + 0.43707093821510296*XNB - 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2)) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.4170990277916469e-20*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 3.5027570743586952e-21*XCR*(1 - XCR)*(-XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 42884.208569999995*XCR*(-XCR - XNB + 1) - 8738.1949664146214*XCR + 42221.999999999993*pow(XNB, 2)*(-XCR - XNB + 1) + 42221.999999999993*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 21503.279314985535*XNB - 29496.435939999992*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 1.0e-13) ? (
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
   )) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17)
   )
   : (
      -6.0229316744766066e-78*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 8738.1949664146177 + 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 181510.70025840754*pow(0.43707093821510296*XCR + 0.43707093821510296*XNB - 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2)))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 3.5027570743586952e-21*XCR*(1 - XCR)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.8295676400933012e-21*XCR*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 42884.208569999995*XCR*(-XCR - XNB + 1) - 8738.1949664146214*XCR + 42221.999999999993*pow(XNB, 2)*(-XCR - XNB + 1) + 42221.999999999993*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 21503.279314985535*XNB - 29496.435939999992*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 1.0e-13) ? (
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
   )) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17)
   )
   : (
      -6.0229316744766066e-78*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 8738.1949664146177 + 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 181510.70025840754*pow(0.43707093821510296*XCR + 0.43707093821510296*XNB - 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2)) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)))*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_CrCr_result;

}

double D_CrNb(double XCR, double XNB) {

   double D_CrNb_result;
   D_CrNb_result = -1.4170990277916469e-20*XCR*XNB*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 42221.999999999993*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 42221.999999999993*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 262253.44338041457*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 296399.45698399993*XNB*(-XCR - XNB + 1) - 275018.52772898547*XNB - 139473.69608899998*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 1.0e-13) ? (
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
   )) + 1) + 275018.52772898547))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.4170990277916469e-20*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 3.5027570743586952e-21*XCR*(1 - XCR)*(-XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 42221.999999999993*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 42221.999999999993*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 262253.44338041457*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 296399.45698399993*XNB*(-XCR - XNB + 1) - 275018.52772898547*XNB - 139473.69608899998*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 1.0e-13) ? (
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
   )) + 1) + 275018.52772898547) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 3.5027570743586952e-21*XCR*(1 - XCR)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.8295676400933012e-21*XCR*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 42221.999999999993*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 42221.999999999993*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 262253.44338041457*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 296399.45698399993*XNB*(-XCR - XNB + 1) - 275018.52772898547*XNB - 139473.69608899998*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 1.0e-13) ? (
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
   )) + 1) + 275018.52772898547))*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_CrNb_result;

}

double D_NbCr(double XCR, double XNB) {

   double D_NbCr_result;
   D_NbCr_result = -3.5027570743586952e-21*XCR*XNB*(-XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 42884.208569999995*XCR*(-XCR - XNB + 1) - 8738.1949664146214*XCR + 42221.999999999993*pow(XNB, 2)*(-XCR - XNB + 1) + 42221.999999999993*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 21503.279314985535*XNB - 29496.435939999992*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 1.0e-13) ? (
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
   )) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17)
   )
   : (
      -6.0229316744766066e-78*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 8738.1949664146177 + 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 181510.70025840754*pow(0.43707093821510296*XCR + 0.43707093821510296*XNB - 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2)))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 3.5027570743586952e-21*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.4170990277916469e-20*XNB*(1 - XNB)*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 42884.208569999995*XCR*(-XCR - XNB + 1) - 8738.1949664146214*XCR + 42221.999999999993*pow(XNB, 2)*(-XCR - XNB + 1) + 42221.999999999993*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 21503.279314985535*XNB - 29496.435939999992*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 1.0e-13) ? (
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
   )) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17)
   )
   : (
      -6.0229316744766066e-78*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 8738.1949664146177 + 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 181510.70025840754*pow(0.43707093821510296*XCR + 0.43707093821510296*XNB - 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2)) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.4170990277916469e-20*XNB*(1 - XNB)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.8295676400933012e-21*XNB*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 42884.208569999995*XCR*(-XCR - XNB + 1) - 8738.1949664146214*XCR + 42221.999999999993*pow(XNB, 2)*(-XCR - XNB + 1) + 42221.999999999993*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 21503.279314985535*XNB - 29496.435939999992*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 1.0e-13) ? (
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
   )) - 19009.441350000005*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 17)
   )
   : (
      -6.0229316744766066e-78*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*pow(3605*XCR + 3605*XNB - 4714, 2)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 8738.1949664146177 + 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 19009.441350000005*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 181510.70025840754*pow(0.43707093821510296*XCR + 0.43707093821510296*XNB - 1, 2)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2)) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)))*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 1.8295676400933012e-21*XNB*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_NbCr_result;

}

double D_NbNb(double XCR, double XNB) {

   double D_NbNb_result;
   D_NbNb_result = -3.5027570743586952e-21*XCR*XNB*(-XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 42221.999999999993*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 42221.999999999993*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 262253.44338041457*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 296399.45698399993*XNB*(-XCR - XNB + 1) - 275018.52772898547*XNB - 139473.69608899998*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 1.0e-13) ? (
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
   )) + 1) + 275018.52772898547) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 3.5027570743586952e-21*XCR*XNB*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XCR)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.4170990277916469e-20*XNB*(1 - XNB)*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 42221.999999999993*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 42221.999999999993*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 262253.44338041457*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 296399.45698399993*XNB*(-XCR - XNB + 1) - 275018.52772898547*XNB - 139473.69608899998*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 1.0e-13) ? (
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
   )) + 1) + 275018.52772898547))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.4170990277916469e-20*XNB*(1 - XNB)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + (1 - XNB)*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.8295676400933012e-21*XNB*(-XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 49246.990599999997*XCR*(-XCR - XNB + 1) - 51010.753158914609*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 247773.60893949994*XNB*(-XCR - XNB + 1) - 63775.837507485521*XNB - 21110.999999999996*pow(-XCR - XNB + 1, 2) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 57393.295333200047 - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 42221.999999999993*pow(XCR, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) + 42221.999999999993*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 262253.44338041457*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 296399.45698399993*XNB*(-XCR - XNB + 1) - 275018.52772898547*XNB - 139473.69608899998*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 1.0e-13) ? (
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
   )) + 1) + 275018.52772898547))*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 1.8295676400933012e-21*XNB*(-XCR - XNB + 1)*(21110.999999999996*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*XCR*pow(XNB, 2) + 21110.999999999996*XCR*XNB*pow(-XCR - XNB + 1, 2) - 21110.999999999996*XCR*XNB*(-XCR - XNB + 1) - 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 8054.3316549999954*XCR*(-XCR - XNB + 1) + 14748.217969999996*XCR*(2*XCR + XNB - 1) - XCR*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 13387.772630000001*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 93138.885483585356*XCR + 21110.999999999996*pow(XNB, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XNB, 2) + 21110.999999999996*XNB*pow(-XCR - XNB + 1, 2) + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 108299.91285049997*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 122646.35932751442*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      0.11111111111111088
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/pow((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1, 2) - 14748.217969999996*(-XCR - XNB + 1)*(2*XCR + XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      log(XCR) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      (-2.7131358180049818*XCR - 2.7131358180049818*XNB + 3.5477731611859875)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 781996027.02054405*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -2.5095548643652528e-79*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      -3127984108.0821762*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 5) - 2.3268190314398873e+27*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 11) - 3.1155389261560675e+45*(633 - 3605*XCR)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) - 8.3881826083407944e+29*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) - 7.0196986429953896e+47*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16)
   )
   : (
      -6.0229316744766066e-78*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 23)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 3.8255060479630414e-47*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 13)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) - 4.3736331682182561e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 3)*(3605*XCR + 3605*XNB - 4714)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         0.11111111111111088
      )
      : (
         1.0
      )) + 9.0469452860367365e-76*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 9.8506780735048315e-45*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 3.9417368928567032e-13*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 46569.442741792685 + 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*(1.9099999999999999*XCR + 1.9099999999999999*XNB - 4.3700000000000001)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 18154.016489250003*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1)) - 23766.893116792686*XCR - 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 8726.0324029999902*XNB*(-XCR - XNB + 1) + 69736.848044499988*XNB*(XCR + 2*XNB - 1) - XNB*(-42221.999999999993*pow(XCR, 2)*XNB*(-XCR - XNB + 1) + 21110.999999999996*pow(XCR, 2)*XNB + 21110.999999999996*pow(XCR, 2)*(-XCR - XNB + 1) - 21110.999999999996*pow(XCR, 2) - 42221.999999999993*XCR*pow(XNB, 2)*(-XCR - XNB + 1) + 21110.999999999996*XCR*pow(XNB, 2) - 42221.999999999993*XCR*XNB*pow(-XCR - XNB + 1, 2) + 42221.999999999993*XCR*XNB*(-XCR - XNB + 1) - 21110.999999999996*XCR*XNB + 21110.999999999996*XCR*pow(-XCR - XNB + 1, 2) + 29496.435939999992*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 19750.554660000002*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 101850.71556008536*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 156925.76089499996*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 300328.32143301441*XNB - 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 69736.848044499988*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) + 19009.441350000005*((XCR > 1.0e-13) ? (
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
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      log(XNB) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) + 19009.441350000005*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1) - 150164.16071650723) - 89153.345075007208*XNB + 9504.7206750000023*(0.52000000000000002 - 1.9099999999999999*XCR)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   ))*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   ))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((XCR > 1.0e-13) ? (
      XCR*log(XCR)
   )
   : (
      0
   )) - 9504.7206750000023*((XNB > 1.0e-13) ? (
      XNB*log(XNB)
   )
   : (
      0
   )) - 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
   )
   : (
      0
   )) + 9504.7206750000023*((XCR + XNB - 1 < -1.0e-13) ? (
      log(-XCR - XNB + 1) + 1
   )
   : (
      0
   )) + 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
      1143.1500000000001/(1201.6666666666699*XCR*(-XCR - XNB + 1) + 580.66666666666697*XCR + 211.0*XNB - 210.99999999900001) < 1
   )
   : (
      1143.1500000000001/(-3605.0*XCR*(-XCR - XNB + 1) - 1742.0*XCR - 633.0*XNB + 633.00000000099999) < 1
   ))) ? (
      781996027.02054405*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4) + 2.3268190314398873e+26*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 10) + 1.9472118288475422e+44*(633 - 3605*XCR)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))/pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 16) + (2.7131358180049818*XCR - 0.47639805070656127)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   )
   : (
      -2.5095548643652528e-79*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 24)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 2.732504319973601e-48*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 14)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) - 1.093408292054564e-16*(633 - 3605*XCR)*pow((-3605*XCR*(-XCR - XNB + 1) - 1742*XCR - 633*XNB + 633)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      )) + 1.0000000000000001e-9, 4)*((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
         -0.33333333333333298
      )
      : (
         1.0
      ))
   ))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
      -0.33333333333333298
   )
   : (
      1.0
   )) + 1) - 9504.7206750000023*((((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 >= 0) ? (
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
   )) + 1))*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_NbNb_result;

}
