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
   xr_gam_Cr_result = 0.52421634830562147 + 2.0284127016252441e-10/r_lav - 2.7082693357358988e-10/r_del;
   return xr_gam_Cr_result;

}

double xr_gam_Nb(double r_del, double r_lav) {

   double xr_gam_Nb_result;
   xr_gam_Nb_result = 0.01299272922003303 - 1.4221829908721655e-11/r_lav + 3.2983057087274643e-11/r_del;
   return xr_gam_Nb_result;

}

double xr_del_Cr(double r_del, double r_lav) {

   double xr_del_Cr_result;
   xr_del_Cr_result = 0.022966218927631978 + 1.6436514881437235e-11/r_lav - 1.7091137328827994e-11/r_del;
   return xr_del_Cr_result;

}

double xr_del_Nb(double r_del, double r_lav) {

   double xr_del_Nb_result;
   xr_del_Nb_result = 0.24984563695705883 + 1.967689408798284e-13/r_lav - 2.5675808729501275e-14/r_del;
   return xr_del_Nb_result;

}

double xr_lav_Cr(double r_del, double r_lav) {

   double xr_lav_Cr_result;
   xr_lav_Cr_result = 0.37392129441013022 + 3.5336079342753527e-11/r_lav - 4.5745621126441332e-11/r_del;
   return xr_lav_Cr_result;

}

double xr_lav_Nb(double r_del, double r_lav) {

   double xr_lav_Nb_result;
   xr_lav_Nb_result = 0.25826261799015571 + 3.2416844758803114e-12/r_lav + 3.3534115774401598e-12/r_del;
   return xr_lav_Nb_result;

}

double inv_fict_det(double pDel, double pGam, double pLav) {

   double inv_fict_det_result;
   inv_fict_det_result = 1.0479034445859148/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171916*pDel*pGam + 0.041785936156220206*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629848*pow(pLav, 2));
   return inv_fict_det_result;

}

double fict_gam_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_gam_Cr_result;
   fict_gam_Cr_result = 0.95428639457822939*INV_DET*(0.016938286205500769*XCR*pDel + 1.0*XCR*pGam + 0.54889980006097583*XCR*pLav - 0.34687466604954664*XNB*pDel - 0.10246420218061991*XNB*pLav + 0.086819224054883931*pow(pDel, 2) + 0.036767802923980242*pDel*pGam + 0.11815012520929236*pDel*pLav - 0.27750258277182049*pGam*pLav - 0.12951476250779617*pow(pLav, 2));
   return fict_gam_Cr_result;

}

double fict_gam_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_gam_Nb_result;
   fict_gam_Nb_result = -0.95428639457822939*INV_DET*(0.0021576593653206372*XCR*pDel + 0.054466450838266962*XCR*pLav - 0.10535187126621839*XNB*pDel - 1.0*XNB*pGam - 0.18138965393196071*XNB*pLav + 0.026258691080522339*pow(pDel, 2) + 0.24849448207756411*pDel*pGam + 0.06992726713680375*pDel*pLav + 0.22257870755542866*pGam*pLav + 0.025258893954068835*pow(pLav, 2));
   return fict_gam_Nb_result;

}

double fict_del_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_del_Cr_result;
   fict_del_Cr_result = 0.33101777443494923*INV_DET*(0.0029867928603642142*XCR*pDel + 0.30371739875396442*XCR*pGam + 0.11224396861282378*XCR*pLav + 1.0*XNB*pGam + 0.15026949298026257*XNB*pLav - 0.25028989589710454*pDel*pGam - 0.038472217853981416*pDel*pLav - 0.10599737173866837*pow(pGam, 2) - 0.38426983287411115*pGam*pLav - 0.074556825343604541*pow(pLav, 2));
   return fict_del_Cr_result;

}

double fict_del_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_del_Nb_result;
   fict_del_Nb_result = 0.23713490337438312*INV_DET*(0.0086829266681549624*XCR*pGam + 0.0010534256500958393*XCR*pLav + 0.0041692787998190662*XNB*pDel + 0.068163631095090937*XNB*pGam + 0.011474488301439057*XNB*pLav + 0.10567112340275649*pDel*pGam + 0.038045328588644858*pDel*pLav + 1.0*pow(pGam, 2) + 0.71270814650507486*pGam*pLav + 0.091137578230976013*pow(pLav, 2));
   return fict_del_Nb_result;

}

double fict_lav_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_lav_Cr_result;
   fict_lav_Cr_result = 0.26481693919946725*INV_DET*(0.010275028791049912*XCR*pDel + 0.65365032685519309*XCR*pGam + 0.33867750853659306*XCR*pLav - 0.18783493715380431*XNB*pDel + 0.36923693162478494*XNB*pGam + 0.048089778433731953*pow(pDel, 2) + 0.054570103782716275*pDel*pGam + 0.093195074562756067*pDel*pLav + 1.0*pow(pGam, 2) + 0.46671552103819908*pGam*pLav);
   return fict_lav_Cr_result;

}

double fict_lav_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav) {

   double fict_lav_Nb_result;
   fict_lav_Nb_result = -0.5238076111848996*INV_DET*(0.00047690026722310252*XCR*pDel - 0.09922840349407383*XCR*pGam - 0.070932051941581878*XNB*pDel - 1.0*XNB*pGam - 0.1712222947342838*XNB*pLav + 0.017223643043877631*pow(pDel, 2) + 0.19525744879034307*pDel*pGam + 0.04125923401282721*pDel*pLav - 0.40549970601319762*pow(pGam, 2) - 0.046017313089315927*pGam*pLav);
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
   g_gam_result = 2267132212.7620249*pow(XCR - 0.52421634830562147, 2) + (15095482346.486225*XCR - 7913298631.5869827)*(XNB - 0.01299272922003303) + 55193083240.685936*pow(XNB - 0.01299272922003303, 2);
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
   dg_gam_dxCr_result = 4534264425.5240498*XCR + 15095482346.486225*XNB - 2573067053.9739881;
   return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB) {

   double dg_gam_dxNb_result;
   dg_gam_dxNb_result = 15095482346.486225*XCR + 110386166481.37187*XNB - 9347516202.3169327;
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
   d2g_gam_dxCrCr_result = 4534264425.5240498;
   return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb() {

   double d2g_gam_dxCrNb_result;
   d2g_gam_dxCrNb_result = 15095482346.486225;
   return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr() {

   double d2g_gam_dxNbCr_result;
   d2g_gam_dxNbCr_result = 15095482346.486225;
   return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb() {

   double d2g_gam_dxNbNb_result;
   d2g_gam_dxNbNb_result = 110386166481.37187;
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

double M_Cr(double XCR, double XNB) {

   double M_Cr_result;
   M_Cr_result = 3.5027570743586952e-21*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR);
   return M_Cr_result;

}

double M_Nb(double XCR, double XNB) {

   double M_Nb_result;
   M_Nb_result = 1.4170990277916469e-20*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB);
   return M_Nb_result;

}

double M_Ni(double XCR, double XNB) {

   double M_Ni_result;
   M_Ni_result = 1.8295676400933012e-21*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return M_Ni_result;

}

double M_CrCr(double XCR, double XNB) {

   double M_CrCr_result;
   M_CrCr_result = 1.4170990277916469e-20*XCR*XNB*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 3.5027570743586952e-21*pow(1 - XCR, 2)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR);
   return M_CrCr_result;

}

double M_CrNb(double XCR, double XNB) {

   double M_CrNb_result;
   M_CrNb_result = -3.5027570743586952e-21*XCR*(1 - XCR)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.4170990277916469e-20*XCR*(1 - XNB)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.8295676400933012e-21*XCR*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return M_CrNb_result;

}

double M_NbCr(double XCR, double XNB) {

   double M_NbCr_result;
   M_NbCr_result = -3.5027570743586952e-21*XNB*(1 - XCR)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.4170990277916469e-20*XNB*(1 - XNB)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.8295676400933012e-21*XNB*(XCR + XNB)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return M_NbCr_result;

}

double M_NbNb(double XCR, double XNB) {

   double M_NbNb_result;
   M_NbNb_result = 3.5027570743586952e-21*XCR*XNB*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.8295676400933012e-21*XNB*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 1.4170990277916469e-20*pow(1 - XNB, 2)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB);
   return M_NbNb_result;

}

double D_CrCr(double XCR, double XNB) {

   double D_CrCr_result;
   D_CrCr_result = 1.4170990277916469e-20*pow(XCR, 2)*XNB*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.8295676400933012e-21*pow(XCR, 2)*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 3.5027570743586952e-21*XCR*pow(1 - XCR, 2)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR);
   return D_CrCr_result;

}

double D_CrNb(double XCR, double XNB) {

   double D_CrNb_result;
   D_CrNb_result = -3.5027570743586952e-21*XCR*XNB*(1 - XCR)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.4170990277916469e-20*XCR*XNB*(1 - XNB)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.8295676400933012e-21*XCR*XNB*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_CrNb_result;

}

double D_NbCr(double XCR, double XNB) {

   double D_NbCr_result;
   D_NbCr_result = -3.5027570743586952e-21*XCR*XNB*(1 - XCR)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.4170990277916469e-20*XCR*XNB*(1 - XNB)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.8295676400933012e-21*XCR*XNB*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
   return D_NbCr_result;

}

double D_NbNb(double XCR, double XNB) {

   double D_NbNb_result;
   D_NbNb_result = 3.5027570743586952e-21*XCR*pow(XNB, 2)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.8295676400933012e-21*pow(XNB, 2)*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB) + 1.4170990277916469e-20*XNB*pow(1 - XNB, 2)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB);
   return D_NbNb_result;

}
