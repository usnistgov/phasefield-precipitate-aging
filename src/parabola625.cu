/******************************************************************************
 *                      Code generated with sympy 1.5.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/
#include "parabola625.cuh"
#include <math.h>

__device__ double d_p(double x)
{

	double p_result;
	p_result = x*x*x*(6.0*x*x - 15.0*x + 10.0);
	return p_result;

}

__device__ double d_pPrime(double x)
{

	double pPrime_result;
	pPrime_result = 30.0*x*x*(1.0 - x)*(1.0 - x);
	return pPrime_result;

}

__device__ double d_interface_profile(double z)
{

	double interface_profile_result;
	interface_profile_result = 1.0/2.0 - 1.0/2.0*tanh(z);
	return interface_profile_result;

}

__device__ double d_kT()
{

	double kT_result;
	kT_result = 1.5782889043500002e-20;
	return kT_result;

}

__device__ double d_RT()
{

	double RT_result;
	RT_result = 9504.6840941999999;
	return RT_result;

}

__device__ double d_Vm()
{

	double Vm_result;
	Vm_result = 1.0000000000000001e-5;
	return Vm_result;

}

__device__ double d_xe_gam_Cr()
{

	double xe_gam_Cr_result;
	xe_gam_Cr_result = 0.52421634830562147;
	return xe_gam_Cr_result;

}

__device__ double d_xe_gam_Nb()
{

	double xe_gam_Nb_result;
	xe_gam_Nb_result = 0.01299272922003303;
	return xe_gam_Nb_result;

}

__device__ double d_xe_del_Cr()
{

	double xe_del_Cr_result;
	xe_del_Cr_result = 0.022966218927631978;
	return xe_del_Cr_result;

}

__device__ double d_xe_del_Nb()
{

	double xe_del_Nb_result;
	xe_del_Nb_result = 0.24984563695705883;
	return xe_del_Nb_result;

}

__device__ double d_xe_lav_Cr()
{

	double xe_lav_Cr_result;
	xe_lav_Cr_result = 0.37392129441013022;
	return xe_lav_Cr_result;

}

__device__ double d_xe_lav_Nb()
{

	double xe_lav_Nb_result;
	xe_lav_Nb_result = 0.25826261799015571;
	return xe_lav_Nb_result;

}

__device__ double d_matrix_min_Cr()
{

	double matrix_min_Cr_result;
	matrix_min_Cr_result = 0.30851614493185742;
	return matrix_min_Cr_result;

}

__device__ double d_matrix_max_Cr()
{

	double matrix_max_Cr_result;
	matrix_max_Cr_result = 0.36263227633734074;
	return matrix_max_Cr_result;

}

__device__ double d_matrix_min_Nb()
{

	double matrix_min_Nb_result;
	matrix_min_Nb_result = 0.019424801579942592;
	return matrix_min_Nb_result;

}

__device__ double d_matrix_max_Nb()
{

	double matrix_max_Nb_result;
	matrix_max_Nb_result = 0.025522709998118311;
	return matrix_max_Nb_result;

}

__device__ double d_enrich_min_Cr()
{

	double enrich_min_Cr_result;
	enrich_min_Cr_result = 0.29783560803725534;
	return enrich_min_Cr_result;

}

__device__ double d_enrich_max_Cr()
{

	double enrich_max_Cr_result;
	enrich_max_Cr_result = 0.35636564894177969;
	return enrich_max_Cr_result;

}

__device__ double d_enrich_min_Nb()
{

	double enrich_min_Nb_result;
	enrich_min_Nb_result = 0.15335241484365611;
	return enrich_min_Nb_result;

}

__device__ double d_enrich_max_Nb()
{

	double enrich_max_Nb_result;
	enrich_max_Nb_result = 0.15955557903581488;
	return enrich_max_Nb_result;

}

__device__ double d_xr_gam_Cr(double r_del, double r_lav)
{

	double xr_gam_Cr_result;
	xr_gam_Cr_result = 0.52421634830562147 + 2.0284127016252441e-10/r_lav - 2.7082693357358988e-10/r_del;
	return xr_gam_Cr_result;

}

__device__ double d_xr_gam_Nb(double r_del, double r_lav)
{

	double xr_gam_Nb_result;
	xr_gam_Nb_result = 0.01299272922003303 - 1.4221829908721655e-11/r_lav + 3.2983057087274643e-11/r_del;
	return xr_gam_Nb_result;

}

__device__ double d_xr_del_Cr(double r_del, double r_lav)
{

	double xr_del_Cr_result;
	xr_del_Cr_result = 0.022966218927631978 + 1.6436514881437235e-11/r_lav - 1.7091137328827994e-11/r_del;
	return xr_del_Cr_result;

}

__device__ double d_xr_del_Nb(double r_del, double r_lav)
{

	double xr_del_Nb_result;
	xr_del_Nb_result = 0.24984563695705883 + 1.967689408798284e-13/r_lav - 2.5675808729501275e-14/r_del;
	return xr_del_Nb_result;

}

__device__ double d_xr_lav_Cr(double r_del, double r_lav)
{

	double xr_lav_Cr_result;
	xr_lav_Cr_result = 0.37392129441013022 + 3.5336079342753527e-11/r_lav - 4.5745621126441332e-11/r_del;
	return xr_lav_Cr_result;

}

__device__ double d_xr_lav_Nb(double r_del, double r_lav)
{

	double xr_lav_Nb_result;
	xr_lav_Nb_result = 0.25826261799015571 + 3.2416844758803114e-12/r_lav + 3.3534115774401598e-12/r_del;
	return xr_lav_Nb_result;

}

__device__ double d_fict_gam_Cr(double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_gam_Cr_result;
	fict_gam_Cr_result = 1.0*(0.016938286205500769*XCR*pDel + 1.0*XCR*pGam + 0.54889980006097594*XCR*pLav - 0.3468746660495467*XNB*pDel - 0.10246420218061991*XNB*pLav + 0.086819224054883945*pow(pDel, 2) + 0.036767802923980249*pDel*pGam + 0.11815012520929236*pDel*pLav - 0.27750258277182049*pGam*pLav - 0.12951476250779617*pow(pLav, 2))/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171917*pDel*pGam + 0.041785936156220213*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629876*pow(pLav, 2));
	return fict_gam_Cr_result;

}

__device__ double d_fict_gam_Nb(double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_gam_Nb_result;
	fict_gam_Nb_result = -1.0*(0.0021576593653206376*XCR*pDel + 0.054466450838266962*XCR*pLav - 0.10535187126621841*XNB*pDel - 1.0*XNB*pGam - 0.18138965393196074*XNB*pLav + 0.026258691080522339*pow(pDel, 2) + 0.24849448207756414*pDel*pGam + 0.06992726713680375*pDel*pLav + 0.22257870755542869*pGam*pLav + 0.025258893954068839*pow(pLav, 2))/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171917*pDel*pGam + 0.041785936156220213*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629876*pow(pLav, 2));
	return fict_gam_Nb_result;

}

__device__ double d_fict_del_Cr(double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_del_Cr_result;
	fict_del_Cr_result = 0.3468746660495467*(0.0029867928603642137*XCR*pDel + 0.30371739875396442*XCR*pGam + 0.11224396861282376*XCR*pLav + 1.0*XNB*pGam + 0.15026949298026254*XNB*pLav - 0.25028989589710454*pDel*pGam - 0.038472217853981416*pDel*pLav - 0.10599737173866837*pow(pGam, 2) - 0.38426983287411109*pGam*pLav - 0.074556825343604527*pow(pLav, 2))/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171917*pDel*pGam + 0.041785936156220213*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629876*pow(pLav, 2));
	return fict_del_Cr_result;

}

__device__ double d_fict_del_Nb(double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_del_Nb_result;
	fict_del_Nb_result = 0.24849448207756414*(0.0086829266681549659*XCR*pGam + 0.0010534256500958396*XCR*pLav + 0.0041692787998190662*XNB*pDel + 0.068163631095090937*XNB*pGam + 0.011474488301439056*XNB*pLav + 0.10567112340275649*pDel*pGam + 0.038045328588644858*pDel*pLav + 1.0*pow(pGam, 2) + 0.71270814650507497*pGam*pLav + 0.091137578230976013*pow(pLav, 2))/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171917*pDel*pGam + 0.041785936156220213*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629876*pow(pLav, 2));
	return fict_del_Nb_result;

}

__device__ double d_fict_lav_Cr(double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_lav_Cr_result;
	fict_lav_Cr_result = 0.27750258277182055*(0.010275028791049912*XCR*pDel + 0.65365032685519298*XCR*pGam + 0.33867750853659306*XCR*pLav - 0.18783493715380425*XNB*pDel + 0.36923693162478488*XNB*pGam + 0.04808977843373196*pow(pDel, 2) + 0.054570103782716275*pDel*pGam + 0.093195074562756067*pDel*pLav + 1.0*pow(pGam, 2) + 0.46671552103819908*pGam*pLav)/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171917*pDel*pGam + 0.041785936156220213*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629876*pow(pLav, 2));
	return fict_lav_Cr_result;

}

__device__ double d_fict_lav_Nb(double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_lav_Nb_result;
	fict_lav_Nb_result = -0.54889980006097605*(0.00047690026722310247*XCR*pDel - 0.099228403494073802*XCR*pGam - 0.070932051941581864*XNB*pDel - 1.0*XNB*pGam - 0.1712222947342838*XNB*pLav + 0.017223643043877624*pow(pDel, 2) + 0.19525744879034301*pDel*pGam + 0.041259234012827196*pDel*pLav - 0.40549970601319757*pow(pGam, 2) - 0.046017313089315913*pGam*pLav)/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171917*pDel*pGam + 0.041785936156220213*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629876*pow(pLav, 2));
	return fict_lav_Nb_result;

}

__device__ double d_s_delta()
{

	double s_delta_result;
	s_delta_result = 0.13;
	return s_delta_result;

}

__device__ double d_s_laves()
{

	double s_laves_result;
	s_laves_result = 0.13;
	return s_laves_result;

}

__device__ double d_GCAL_gam(double XCR, double XNB)
{

	double GCAL_gam_result;
	GCAL_gam_result = -2111099999.9999998*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999995*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 669388631.5*XCR*(-XCR - XNB + 1) + 1707300680.1792686*XCR + 6973684804.4499989*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 7846288044.7499981*XNB*(-XCR - XNB + 1) + 1069046462.7507229*XNB + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
	                      XCR*log(XCR)
	                  )
	                  : (
	                      0
	                  )) + 950472067.50000012*((XNB > 9.9999999999999998e-17) ? (
	                              XNB*log(XNB)
	                          )
	                          : (
	                              0
	                          )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                      (-XCR - XNB + 1)*log(-XCR - XNB + 1)
	                                  )
	                                  : (
	                                      0
	                                  )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                          -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                          )
	                                          : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                  2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                  )
	                                                  : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                          -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                          )
	                                                          : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                  -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                  )
	                                                                  : (
	                                                                          0
	                                                                  )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                          -0.33333333333333298
	                                                                          )
	                                                                          : (
	                                                                                  1.0
	                                                                          )) + 1) - 5464277694.8364201;
	return GCAL_gam_result;

}

__device__ double d_GCAL_del(double XCR, double XNB)
{

	double GCAL_del_result;
	GCAL_del_result = -21668797081.409531*XCR*XNB - 5258769591.26929*XCR*(1 - 4*XNB) - 1231849999.9999998*XNB*(1 - 1.3333333333333333*XCR)*(1 - 4*XNB) - 34242601639.394947*XNB*(1 - 1.3333333333333333*XCR) - 4964277694.8364201*(1 - 1.3333333333333333*XCR)*(1 - 4*XNB) + 712854050.625*((1.3333333333333333*XCR > 9.9999999999999998e-17) ? (
	                      1.3333333333333333*XCR*log(1.3333333333333333*XCR)
	                  )
	                  : (
	                      0
	                  )) + 237618016.87500003*((4*XNB > 9.9999999999999998e-17) ? (
	                              4*XNB*log(4*XNB)
	                          )
	                          : (
	                              0
	                          )) + 712854050.625*((1.3333333333333333*XCR - 1 < -9.9999999999999998e-17) ? (
	                                      (1 - 1.3333333333333333*XCR)*log(1 - 1.3333333333333333*XCR)
	                                  )
	                                  : (
	                                      0
	                                  )) + 237618016.87500003*((4*XNB - 1 < -9.9999999999999998e-17) ? (
	                                          (1 - 4*XNB)*log(1 - 4*XNB)
	                                          )
	                                          : (
	                                                  0
	                                          ));
	return GCAL_del_result;

}

__device__ double d_GCAL_lav(double XCR, double XNB)
{

	double GCAL_lav_result;
	GCAL_lav_result = -46695351257.249992*XNB*(1 - 3*XNB)*(-1.5*XCR - 1.5*XNB + 1.5) + 1851135999.9999998*XNB*(1 - 3*XNB)*(1.5*XCR + 1.5*XNB - 0.5) - 10298680536.599998*XNB*(-1.5*XCR - 1.5*XNB + 1.5)*(1.5*XCR + 1.5*XNB - 0.5) - 22164936866.458534*XNB*(-1.5*XCR - 1.5*XNB + 1.5) - 16694022288.40456*XNB*(1.5*XCR + 1.5*XNB - 0.5) + 4855811416.8900013*(1 - 3*XNB)*(-1.5*XCR - 1.5*XNB + 1.5) - 4004010359.6571507*(1 - 3*XNB)*(1.5*XCR + 1.5*XNB - 0.5) + 316824022.5*((3*XNB > 9.9999999999999998e-17) ? (
	                      3*XNB*log(3*XNB)
	                  )
	                  : (
	                      0
	                  )) + 316824022.5*((3*XNB - 1 < -9.9999999999999998e-17) ? (
	                                        (1 - 3*XNB)*log(1 - 3*XNB)
	                                    )
	                                    : (
	                                        0
	                                    )) + 633648045.0*((1.5*XCR + 1.5*XNB - 1.5 < -9.9999999999999998e-17) ? (
	                                            (-1.5*XCR - 1.5*XNB + 1.5)*log(-1.5*XCR - 1.5*XNB + 1.5)
	                                            )
	                                            : (
	                                                    0
	                                            )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 9.9999999999999998e-17) ? (
	                                                    (1.5*XCR + 1.5*XNB - 0.5)*log(1.5*XCR + 1.5*XNB - 0.5)
	                                                    )
	                                                    : (
	                                                            0
	                                                    ));
	return GCAL_lav_result;

}

__device__ double d_g_gam(double XCR, double XNB)
{

	double g_gam_result;
	g_gam_result = 2267132212.7620249*pow(XCR - 0.52421634830562147, 2) + (15095482346.486225*XCR - 7913298631.5869827)*(XNB - 0.01299272922003303) + 55193083240.685936*pow(XNB - 0.01299272922003303, 2);
	return g_gam_result;

}

__device__ double d_g_del(double XCR, double XNB)
{

	double g_del_result;
	g_del_result = 21346492990.798885*pow(XCR - 0.022966218927631978, 2) + (16906497386.287418*XCR - 388278320.27291465)*(XNB - 0.24984563695705883) + 3085369132931.8848*pow(XNB - 0.24984563695705883, 2);
	return g_del_result;

}

__device__ double d_g_lav(double XCR, double XNB)
{

	double g_lav_result;
	g_lav_result = 8866730284.8069954*pow(XCR - 0.37392129441013022, 2) + (24191004361.532181*XCR - 9045531663.945219)*(XNB - 0.25826261799015571) + 98294310279.883911*pow(XNB - 0.25826261799015571, 2);
	return g_lav_result;

}

__device__ double d_dGCAL_gam_dxCr(double XCR, double XNB)
{

	double dGCAL_gam_dxCr_result;
	dGCAL_gam_dxCr_result = 2111099999.9999998*pow(XCR, 2)*XNB + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) + 2949643593.999999*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 1338777263.0*XCR - 2111099999.9999998*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 6973684804.4499989*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 8515676676.2499981*XNB + (-1474821796.9999995*XCR - 1474821796.9999995*XNB + 1474821796.9999995)*(2*XCR + XNB - 1) + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
	                            log(XCR) + 1
	                        )
	                        : (
	                            0
	                        )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                    -log(-XCR - XNB + 1) - 1
	                                )
	                                : (
	                                    0
	                                )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                        1.8087572120033193*XCR + 0.90437860600165965*XNB - 0.15022120760294841*(6*XCR + 3*XNB - 16041.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(18*XCR + 9*XNB - 48123.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(30*XCR + 15*XNB - 16041.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 1.3413904039641813
	                                        )
	                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                -5.4262716360099637*XCR - 2.7131358180049818*XNB - 5.7509938641697477e-11*(-30*XCR - 15*XNB + 16041.0/721.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) - 2.5140428542154968e-7*(-18*XCR - 9*XNB + 48123.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 0.0055637484297388446*(-6*XCR - 3*XNB + 16041.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.0241712118925488
	                                                )
	                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                        -13.314924373336252*(10*XCR + 5*XNB - 5347.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(30*XCR + 15*XNB - 16041.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(50*XCR + 25*XNB - 26735.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                        )
	                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                -0.00099141855897878122*(-50*XCR - 25*XNB + 26735.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24) - 0.0028656939921696892*(-30*XCR - 15*XNB + 16041.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.054793927462289095*(-10*XCR - 5*XNB + 5347.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4)
	                                                                )
	                                                                : (
	                                                                        0
	                                                                )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                        -0.33333333333333298
	                                                                        )
	                                                                        : (
	                                                                                1.0
	                                                                        )) + 1) + 1037912048.6792686 + 950472067.50000012*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                -0.33333333333333298
	                                                                                )
	                                                                                : (
	                                                                                        1.0
	                                                                                ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                        -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                        )
	                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                )
	                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                        -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                        )
	                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                )
	                                                                                                                : (
	                                                                                                                        0
	                                                                                                                )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                        -0.33333333333333298
	                                                                                                                        )
	                                                                                                                        : (
	                                                                                                                                1.0
	                                                                                                                        )) + 1);
	return dGCAL_gam_dxCr_result;

}

__device__ double d_dGCAL_gam_dxNb(double XCR, double XNB)
{

	double dGCAL_gam_dxNb_result;
	dGCAL_gam_dxNb_result = 2111099999.9999998*pow(XCR, 2)*XNB - 2111099999.9999998*pow(XCR, 2)*(-XCR - XNB + 1) + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) - 2111099999.9999998*XCR*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999995*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 8515676676.2499981*XCR + 13947369608.899998*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 15692576089.499996*XNB + 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                            -0.33333333333333298
	                        )
	                        : (
	                            1.0
	                        ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                            )
	                            : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                   2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                               )
	                               : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                      -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                  )
	                                  : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                          -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                     )
	                                     : (
	                                             0
	                                     )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                             -0.33333333333333298
	                                             )
	                                             : (
	                                                     1.0
	                                             )) + 1) + (-6973684804.4499989*XCR - 6973684804.4499989*XNB + 6973684804.4499989)*(XCR + 2*XNB - 1) + 950472067.50000012*((XNB > 9.9999999999999998e-17) ? (
	                                                     log(XNB) + 1
	                                                     )
	                                                     : (
	                                                             0
	                                                     )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                             -log(-XCR - XNB + 1) - 1
	                                                             )
	                                                             : (
	                                                                     0
	                                                             )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                     0.90437860600165965*XCR - 0.15022120760294841*(3*XCR - 1899.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(9*XCR - 5697.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(15*XCR - 1899.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 0.15879935023552028
	                                                                     )
	                                                                     : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                             -2.7131358180049818*XCR - 0.0055637484297388446*(1899.0/3605.0 - 3*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 2.5140428542154968e-7*(5697.0/3605.0 - 9*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 5.7509938641697477e-11*(1899.0/721.0 - 15*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + 0.47639805070656127
	                                                                             )
	                                                                             : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                     -13.314924373336252*(5*XCR - 633.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(15*XCR - 1899.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(25*XCR - 3165.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                                                     )
	                                                                                     : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                             -0.054793927462289095*(633.0/721.0 - 5*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0028656939921696892*(1899.0/721.0 - 15*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.00099141855897878122*(3165.0/721.0 - 25*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24)
	                                                                                             )
	                                                                                             : (
	                                                                                                     0
	                                                                                             )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                     -0.33333333333333298
	                                                                                                     )
	                                                                                                     : (
	                                                                                                             1.0
	                                                                                                     )) + 1) - 6777241581.9992752;
	return dGCAL_gam_dxNb_result;

}

__device__ double d_dGCAL_del_dxCr(double XCR, double XNB)
{

	double dGCAL_del_dxCr_result;
	dGCAL_del_dxCr_result = 1642466666.6666663*XNB*(1 - 4*XNB) + 18546935763.733315*XNB + 712854050.625*((1.3333333333333333*XCR > 9.9999999999999998e-17) ? (
	                            1.3333333333333333*log(1.3333333333333333*XCR) + 1.3333333333333333
	                        )
	                        : (
	                            0
	                        )) + 712854050.625*((1.3333333333333333*XCR - 1 < -9.9999999999999998e-17) ? (
	                                    -1.3333333333333333*log(1 - 1.3333333333333333*XCR) - 1.3333333333333333
	                                )
	                                : (
	                                    0
	                                )) + 1360267335.1792698;
	return dGCAL_del_dxCr_result;

}

__device__ double d_dGCAL_del_dxNb(double XCR, double XNB)
{

	double dGCAL_del_dxNb_result;
	dGCAL_del_dxNb_result = 18546935763.733315*XCR + 4927399999.999999*XNB*(1 - 1.3333333333333333*XCR) + (1 - 1.3333333333333333*XCR)*(4927399999.999999*XNB - 1231849999.9999998) + 237618016.87500003*((4*XNB > 9.9999999999999998e-17) ? (
	                            4*log(4*XNB) + 4
	                        )
	                        : (
	                            0
	                        )) + 237618016.87500003*((4*XNB - 1 < -9.9999999999999998e-17) ? (
	                                    -4*log(1 - 4*XNB) - 4
	                                )
	                                : (
	                                    0
	                                )) - 14385490860.049267;
	return dGCAL_del_dxNb_result;

}

__device__ double d_dGCAL_lav_dxCr(double XCR, double XNB)
{

	double dGCAL_lav_dxCr_result;
	dGCAL_lav_dxCr_result = 72819730885.874985*XNB*(1 - 3*XNB) - 15448020804.899998*XNB*(-1.5*XCR - 1.5*XNB + 1.5) + 15448020804.899998*XNB*(1.5*XCR + 1.5*XNB - 0.5) + 48075569861.543144*XNB + 633648045.0*((1.5*XCR + 1.5*XNB - 1.5 < -9.9999999999999998e-17) ? (
	                            -1.5*log(-1.5*XCR - 1.5*XNB + 1.5) - 1.5
	                        )
	                        : (
	                            0
	                        )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 9.9999999999999998e-17) ? (
	                                    1.5*log(1.5*XCR + 1.5*XNB - 0.5) + 1.5
	                                )
	                                : (
	                                    0
	                                )) - 13289732664.820728;
	return dGCAL_lav_dxCr_result;

}

__device__ double d_dGCAL_lav_dxNb(double XCR, double XNB)
{

	double dGCAL_lav_dxNb_result;
	dGCAL_lav_dxNb_result = 48075569861.543152*XCR + 72819730885.874985*XNB*(1 - 3*XNB) + 124638032966.84998*XNB*(-1.5*XCR - 1.5*XNB + 1.5) + 9894612804.8999977*XNB*(1.5*XCR + 1.5*XNB - 0.5) + 96151139723.086304*XNB + (1851135999.9999998 - 5553407999.999999*XNB)*(1.5*XCR + 1.5*XNB - 0.5) + (140086053771.74997*XNB - 46695351257.249992)*(-1.5*XCR - 1.5*XNB + 1.5) + (1.5*XCR + 1.5*XNB - 0.5)*(15448020804.899998*XCR + 15448020804.899998*XNB - 15448020804.899998) + 316824022.5*((3*XNB > 9.9999999999999998e-17) ? (
	                            3*log(3*XNB) + 3
	                        )
	                        : (
	                            0
	                        )) + 316824022.5*((3*XNB - 1 < -9.9999999999999998e-17) ? (
	                                    -3*log(1 - 3*XNB) - 3
	                                )
	                                : (
	                                    0
	                                )) + 633648045.0*((1.5*XCR + 1.5*XNB - 1.5 < -9.9999999999999998e-17) ? (
	                                        -1.5*log(-1.5*XCR - 1.5*XNB + 1.5) - 1.5
	                                        )
	                                        : (
	                                                0
	                                        )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 9.9999999999999998e-17) ? (
	                                                1.5*log(1.5*XCR + 1.5*XNB - 0.5) + 1.5
	                                                )
	                                                : (
	                                                        0
	                                                )) - 66047293735.796982;
	return dGCAL_lav_dxNb_result;

}

__device__ double d_dg_gam_dxCr(double XCR, double XNB)
{

	double dg_gam_dxCr_result;
	dg_gam_dxCr_result = 4534264425.5240498*XCR + 15095482346.486225*XNB - 2573067053.9739881;
	return dg_gam_dxCr_result;

}

__device__ double d_dg_gam_dxNb(double XCR, double XNB)
{

	double dg_gam_dxNb_result;
	dg_gam_dxNb_result = 15095482346.486225*XCR + 110386166481.37187*XNB - 9347516202.3169327;
	return dg_gam_dxNb_result;

}

__device__ double d_dg_del_dxCr(double XCR, double XNB)
{

	double dg_del_dxCr_result;
	dg_del_dxCr_result = 42692985981.597771*XCR + 16906497386.287418*XNB - 5204511070.9175282;
	return dg_del_dxCr_result;

}

__device__ double d_dg_del_dxNb(double XCR, double XNB)
{

	double dg_del_dxNb_result;
	dg_del_dxNb_result = 16906497386.287418*XCR + 6170738265863.7695*XNB - 1542120310850.303;
	return dg_del_dxNb_result;

}

__device__ double d_dg_lav_dxCr(double XCR, double XNB)
{

	double dg_lav_dxCr_result;
	dg_lav_dxCr_result = 17733460569.613991*XCR + 24191004361.532181*XNB - 12878550648.781645;
	return dg_lav_dxCr_result;

}

__device__ double d_dg_lav_dxNb(double XCR, double XNB)
{

	double dg_lav_dxNb_result;
	dg_lav_dxNb_result = 24191004361.532181*XCR + 196588620559.76782*XNB - 59817023476.78421;
	return dg_lav_dxNb_result;

}

__device__ double d_d2GCAL_gam_dxCrCr(double XCR, double XNB)
{

	double d2GCAL_gam_dxCrCr_result;
	d2GCAL_gam_dxCrCr_result = 4222199999.9999995*XCR*XNB - 17697861563.999992*XCR + 4222199999.9999995*pow(XNB, 2) - 4222199999.9999995*XNB*(XCR + XNB - 1) - 22796300390.899994*XNB + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
	                               1.0/XCR
	                           )
	                           : (
	                               0
	                           )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                       -1/(XCR + XNB - 1)
	                                   )
	                                   : (
	                                       0
	                                   )) + 950472067.50000012*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                           1.3870814277714769e-7*pow(7210*XCR + 3605*XNB - 5347, 2)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5) + 3.4268566696025357e-8*pow(7210*XCR + 3605*XNB - 5347, 2)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 11) + 1.523920911778034e-8*pow(7210*XCR + 3605*XNB - 5347, 2)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 17) + 1.8087572120033193 - 0.9013272456176904/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 0.089071029899142604/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 0.024756142834362654/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16)
	                                           )
	                                           : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                   -5.1373386213758545e-9*pow(7210*XCR + 3605*XNB - 5347, 2)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5) - 1.7410235581987164e-12*pow(7210*XCR + 3605*XNB - 5347, 2)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 11) - 1.0620466853524362e-15*pow(7210*XCR + 3605*XNB - 5347, 2)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 17) - 5.4262716360099637 + 0.033382490578433066/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.5252771375878952e-6/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.7252981592509244e-9/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16)
	                                                   )
	                                                   : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                           -1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3)*(4.7564684341884595e+79*XCR*(XCR + XNB - 1) - 2.2984099895579189e+79*XCR - 8.3518571951214843e+78*XNB + 1.3853995416615124e+82*pow(7210*XCR + 3605*XNB - 5347, 2)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 2.3735768934135431e+77*pow(7210*XCR + 3605*XNB - 5347, 2)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 7.3198819395753102e+72*pow(7210*XCR + 3605*XNB - 5347, 2) + 1.5003905898684632e+88*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 21) + 4.4067235208892496e+83*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 11) + 8.3518571951346796e+78)
	                                                           )
	                                                           : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                   (1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3)*(1.957394417361506e+77*XCR*(XCR + XNB - 1) - 9.4584773232836154e+76*XCR - 3.4369782695973187e+76*XNB + 1.6350975644205316e+70*pow(7210*XCR + 3605*XNB - 5347, 2)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.654186547737425e+70*pow(7210*XCR + 3605*XNB - 5347, 2)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 3.0122970944754354e+70*pow(7210*XCR + 3605*XNB - 5347, 2) + 1.7708140687206949e+76*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 21) + 3.0711213898656102e+76*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 11) + 3.4369782695810295e+76)
	                                                                   )
	                                                                   : (
	                                                                           0
	                                                                   )))))*log(-(-1.9099999999999999*XCR*(XCR + XNB - 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                           -0.33333333333333298
	                                                                           )
	                                                                           : (
	                                                                                   1.0
	                                                                           )) + 1) + 10187708044.999996 + 1900944135.0000002*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                   -0.33333333333333298
	                                                                                   )
	                                                                                   : (
	                                                                                           1.0
	                                                                                   ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                           1.8087572120033193*XCR + 0.90437860600165965*XNB - 1.0/721.0*(17.849178983575474*XCR + 8.9245894917877369*XNB - 13.237109573533711)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16) - 1.0/3605.0*(321.10106278640905*XCR + 160.55053139320452*XNB - 238.13139843535771)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 1.0/3605.0*(3249.2847204517739*XCR + 1624.6423602258869*XNB - 2409.6983911588954)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 1.3413904039641813
	                                                                                           )
	                                                                                           : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                   -5.4262716360099637*XCR - 2.7131358180049818*XNB + (1.0/721.0)*(1.2439399728199165e-6*XCR + 6.2196998640995823e-7*XNB - 9.2251692575146925e-7)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + (1.0/3605.0)*(0.016313624081004358*XCR + 0.0081568120405021791*XNB - 0.012098328427341236)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + (1.0/3605.0)*(120.34387853525121*XCR + 60.171939267625604*XNB - 89.248088561440795)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.0241712118925488
	                                                                                                   )
	                                                                                                   : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                           -1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0*(7210*XCR + 3605*XNB - 5347)*(5.7724980902563013e+80*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 20) + 1.6954120667239596e+76*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 4)
	                                                                                                           )
	                                                                                                           : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                   (1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0)*(7210*XCR + 3605*XNB - 5347)*(6.8129065184188816e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.1815618198124465e+69*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4)
	                                                                                                                   )
	                                                                                                                   : (
	                                                                                                                           0
	                                                                                                                   )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                           -0.33333333333333298
	                                                                                                                           )
	                                                                                                                           : (
	                                                                                                                                   1.0
	                                                                                                                           )) + 1) + 3630803297.8500004*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                   -0.33333333333333298
	                                                                                                                                   )
	                                                                                                                                   : (
	                                                                                                                                           1.0
	                                                                                                                                   ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                                           0.90437860600165965*XCR*(XCR + XNB - 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 + 0.15022120760294844/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3) + 0.0049483905499523653/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 9) + 0.00082520476114542193/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 15)
	                                                                                                                                           )
	                                                                                                                                           : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                                                   -2.7131358180049818*XCR*(XCR + XNB - 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697484e-11/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                   )
	                                                                                                                                                   : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                                           -1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0*(1.1544996180512604e+80*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 5.6513735557465315e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                           )
	                                                                                                                                                           : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                                                   (1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0)*(1.3625813036837763e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 3.9385393993748223e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                                                                                                                                                   )
	                                                                                                                                                                   : (
	                                                                                                                                                                           0
	                                                                                                                                                                   )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                           -0.33333333333333298
	                                                                                                                                                                           )
	                                                                                                                                                                           : (
	                                                                                                                                                                                   1.0
	                                                                                                                                                                           )) + 1) - 22727783125.266747*pow(0.78118609406952966*XCR + 0.39059304703476483*XNB - 1, 2)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                   0.11111111111111088
	                                                                                                                                                                                   )
	                                                                                                                                                                                   : (
	                                                                                                                                                                                           1.0
	                                                                                                                                                                                   ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                                                                                           0.90437860600165965*XCR*(XCR + XNB - 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 + 0.15022120760294844/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3) + 0.0049483905499523653/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 9) + 0.00082520476114542193/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 15)
	                                                                                                                                                                                           )
	                                                                                                                                                                                           : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                                                                                                   -2.7131358180049818*XCR*(XCR + XNB - 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697484e-11/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                                                                   )
	                                                                                                                                                                                                   : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                                                                                           -1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0*(1.1544996180512604e+80*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 5.6513735557465315e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                                                           )
	                                                                                                                                                                                                           : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                                                                                                   (1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0)*(1.3625813036837763e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 3.9385393993748223e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                                                                                                                                                                                                   )
	                                                                                                                                                                                                                   : (
	                                                                                                                                                                                                                           0
	                                                                                                                                                                                                                   )))))/pow((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                                                           -0.33333333333333298
	                                                                                                                                                                                                                           )
	                                                                                                                                                                                                                           : (
	                                                                                                                                                                                                                                   1.0
	                                                                                                                                                                                                                           )) + 1, 2);
	return d2GCAL_gam_dxCrCr_result;

}

__device__ double d_d2GCAL_gam_dxCrNb(double XCR, double XNB)
{

	double d2GCAL_gam_dxCrNb_result;
	d2GCAL_gam_dxCrNb_result = 2111099999.9999998*pow(XCR, 2) + 4222199999.9999995*XCR*XNB - 22796300390.899994*XCR + 2111099999.9999998*pow(XNB, 2) - 44791752420.699997*XNB + 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                               -0.33333333333333298
	                           )
	                           : (
	                               1.0
	                           ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                   1.8087572120033193*XCR + 0.90437860600165965*XNB - 1.0/721.0*(17.849178983575474*XCR + 8.9245894917877369*XNB - 13.237109573533711)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16) - 1.0/3605.0*(321.10106278640905*XCR + 160.55053139320452*XNB - 238.13139843535771)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 1.0/3605.0*(3249.2847204517739*XCR + 1624.6423602258869*XNB - 2409.6983911588954)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 1.3413904039641813
	                               )
	                               : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                      -5.4262716360099637*XCR - 2.7131358180049818*XNB + (1.0/721.0)*(1.2439399728199165e-6*XCR + 6.2196998640995823e-7*XNB - 9.2251692575146925e-7)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + (1.0/3605.0)*(0.016313624081004358*XCR + 0.0081568120405021791*XNB - 0.012098328427341236)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + (1.0/3605.0)*(120.34387853525121*XCR + 60.171939267625604*XNB - 89.248088561440795)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.0241712118925488
	                                  )
	                                  : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                          -1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0*(7210*XCR + 3605*XNB - 5347)*(5.7724980902563013e+80*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 20) + 1.6954120667239596e+76*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 4)
	                                     )
	                                     : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                             (1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0)*(7210*XCR + 3605*XNB - 5347)*(6.8129065184188816e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.1815618198124465e+69*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4)
	                                        )
	                                        : (
	                                                0
	                                        )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                -0.33333333333333298
	                                                )
	                                                : (
	                                                        1.0
	                                                )) + 1) - 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                        0.11111111111111088
	                                                        )
	                                                        : (
	                                                                1.0
	                                                        ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                0.90437860600165965*XCR*(XCR + XNB - 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 + 0.15022120760294844/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3) + 0.0049483905499523653/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 9) + 0.00082520476114542193/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 15)
	                                                                )
	                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                        -2.7131358180049818*XCR*(XCR + XNB - 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697484e-11/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                        )
	                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                -1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0*(1.1544996180512604e+80*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 5.6513735557465315e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                )
	                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                        (1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0)*(1.3625813036837763e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 3.9385393993748223e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                                                                        )
	                                                                                        : (
	                                                                                                0
	                                                                                        )))))/pow((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                -0.33333333333333298
	                                                                                                )
	                                                                                                : (
	                                                                                                        1.0
	                                                                                                )) + 1, 2) - 2111099999.9999998*(XCR + XNB - 1)*(XCR + XNB - 1) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                        -1/(XCR + XNB - 1)
	                                                                                                        )
	                                                                                                        : (
	                                                                                                                0
	                                                                                                        )) + 950472067.50000012*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                1.3870814277714769e-7*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5) + 3.4268566696025357e-8*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 11) + 1.523920911778034e-8*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 17) + 0.90437860600165965 - 0.4506636228088452/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 0.044535514949571302/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 0.012378071417181327/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16)
	                                                                                                                )
	                                                                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                        -5.1373386213758545e-9*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5) - 1.7410235581987164e-12*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 11) - 1.0620466853524362e-15*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 17) - 2.7131358180049818 + 0.016691245289216533/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 2.2626385687939476e-6/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 8.6264907962546218e-10/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16)
	                                                                                                                        )
	                                                                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                -1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3)*(2.3782342170942297e+79*XCR*(XCR + XNB - 1) - 1.1492049947789595e+79*XCR - 4.1759285975607422e+78*XNB + 1.3853995416615124e+82*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 2.3735768934135431e+77*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 7.3198819395753102e+72*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347) + 7.5019529493423158e+87*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 21) + 2.2033617604446248e+83*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 11) + 4.1759285975673398e+78)
	                                                                                                                                )
	                                                                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                        (1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3)*(9.7869720868075301e+76*XCR*(XCR + XNB - 1) - 4.7292386616418077e+76*XCR - 1.7184891347986593e+76*XNB + 1.6350975644205316e+70*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.654186547737425e+70*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 3.0122970944754354e+70*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347) + 8.8540703436034745e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 21) + 1.5355606949328051e+76*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 11) + 1.7184891347905147e+76)
	                                                                                                                                        )
	                                                                                                                                        : (
	                                                                                                                                                0
	                                                                                                                                        )))))*log(-(-1.9099999999999999*XCR*(XCR + XNB - 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                -0.33333333333333298
	                                                                                                                                                )
	                                                                                                                                                : (
	                                                                                                                                                        1.0
	                                                                                                                                                )) + 1) + 25412689879.149994 + 950472067.50000012*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                        )
	                                                                                                                                                        : (
	                                                                                                                                                                1.0
	                                                                                                                                                        ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                                                                0.90437860600165965*XCR - 1.0/721.0*(8.9245894917877369*XCR - 1.5670638414151561)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16) - 1.0/3605.0*(160.55053139320452*XCR - 28.19098096307863)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 1.0/3605.0*(1624.6423602258869*XCR - 285.27007323799899)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 0.15879935023552028
	                                                                                                                                                                )
	                                                                                                                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                                                                        -2.7131358180049818*XCR + (1.0/721.0)*(6.2196998640995823e-7*XCR - 1.0921137348058351e-7)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + (1.0/3605.0)*(0.0081568120405021791*XCR - 0.0014322502140465686)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + (1.0/3605.0)*(60.171939267625604*XCR - 10.565558268074065)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 0.47639805070656127
	                                                                                                                                                                        )
	                                                                                                                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                                                                -1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0*(3605*XCR - 633)*(5.7724980902563013e+80*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 20) + 1.6954120667239596e+76*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 4)
	                                                                                                                                                                                )
	                                                                                                                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                                                                        (1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0)*(3605*XCR - 633)*(6.8129065184188816e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.1815618198124465e+69*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4)
	                                                                                                                                                                                        )
	                                                                                                                                                                                        : (
	                                                                                                                                                                                                0
	                                                                                                                                                                                        )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                                -0.33333333333333298
	                                                                                                                                                                                                )
	                                                                                                                                                                                                : (
	                                                                                                                                                                                                        1.0
	                                                                                                                                                                                                )) + 1) + 1815401648.9250002*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                                                                        )
	                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                1.0
	                                                                                                                                                                                                        ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                                                                                                                0.90437860600165965*XCR*(XCR + XNB - 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 + 0.15022120760294844/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3) + 0.0049483905499523653/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 9) + 0.00082520476114542193/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 15)
	                                                                                                                                                                                                                )
	                                                                                                                                                                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                                                                                                                        -2.7131358180049818*XCR*(XCR + XNB - 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697484e-11/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                                                                                                                -1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0*(1.1544996180512604e+80*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 5.6513735557465315e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                                                                                                                        (1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0)*(1.3625813036837763e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 3.9385393993748223e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                0
	                                                                                                                                                                                                                                        )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                                                                                -0.33333333333333298
	                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                        1.0
	                                                                                                                                                                                                                                                )) + 1);
	return d2GCAL_gam_dxCrNb_result;

}

__device__ double d_d2GCAL_gam_dxNbCr(double XCR, double XNB)
{

	double d2GCAL_gam_dxNbCr_result;
	d2GCAL_gam_dxNbCr_result = 2111099999.9999998*pow(XCR, 2) + 4222199999.9999995*XCR*XNB - 22796300390.899994*XCR + 2111099999.9999998*pow(XNB, 2) - 44791752420.699997*XNB + 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                               -0.33333333333333298
	                           )
	                           : (
	                               1.0
	                           ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                   1.8087572120033193*XCR + 0.90437860600165965*XNB - 1.0/721.0*(17.849178983575474*XCR + 8.9245894917877369*XNB - 13.237109573533711)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16) - 1.0/3605.0*(321.10106278640905*XCR + 160.55053139320452*XNB - 238.13139843535771)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 1.0/3605.0*(3249.2847204517739*XCR + 1624.6423602258869*XNB - 2409.6983911588954)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 1.3413904039641813
	                               )
	                               : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                      -5.4262716360099637*XCR - 2.7131358180049818*XNB + (1.0/721.0)*(1.2439399728199165e-6*XCR + 6.2196998640995823e-7*XNB - 9.2251692575146925e-7)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + (1.0/3605.0)*(0.016313624081004358*XCR + 0.0081568120405021791*XNB - 0.012098328427341236)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + (1.0/3605.0)*(120.34387853525121*XCR + 60.171939267625604*XNB - 89.248088561440795)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.0241712118925488
	                                  )
	                                  : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                          -1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0*(7210*XCR + 3605*XNB - 5347)*(5.7724980902563013e+80*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 20) + 1.6954120667239596e+76*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 4)
	                                     )
	                                     : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                             (1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0)*(7210*XCR + 3605*XNB - 5347)*(6.8129065184188816e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.1815618198124465e+69*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4)
	                                        )
	                                        : (
	                                                0
	                                        )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                -0.33333333333333298
	                                                )
	                                                : (
	                                                        1.0
	                                                )) + 1) - 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                        0.11111111111111088
	                                                        )
	                                                        : (
	                                                                1.0
	                                                        ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                0.90437860600165965*XCR*(XCR + XNB - 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 + 0.15022120760294844/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3) + 0.0049483905499523653/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 9) + 0.00082520476114542193/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 15)
	                                                                )
	                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                        -2.7131358180049818*XCR*(XCR + XNB - 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697484e-11/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                        )
	                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                -1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0*(1.1544996180512604e+80*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 5.6513735557465315e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                )
	                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                        (1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0)*(1.3625813036837763e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 3.9385393993748223e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                                                                        )
	                                                                                        : (
	                                                                                                0
	                                                                                        )))))/pow((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                -0.33333333333333298
	                                                                                                )
	                                                                                                : (
	                                                                                                        1.0
	                                                                                                )) + 1, 2) - 2111099999.9999998*(XCR + XNB - 1)*(XCR + XNB - 1) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                        -1/(XCR + XNB - 1)
	                                                                                                        )
	                                                                                                        : (
	                                                                                                                0
	                                                                                                        )) + 950472067.50000012*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                1.3870814277714769e-7*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5) + 3.4268566696025357e-8*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 11) + 1.523920911778034e-8*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 17) + 0.90437860600165965 - 0.4506636228088452/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 0.044535514949571302/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 0.012378071417181327/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16)
	                                                                                                                )
	                                                                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                        -5.1373386213758545e-9*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5) - 1.7410235581987164e-12*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 11) - 1.0620466853524362e-15*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 17) - 2.7131358180049818 + 0.016691245289216533/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 2.2626385687939476e-6/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 8.6264907962546218e-10/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16)
	                                                                                                                        )
	                                                                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                -1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3)*(2.3782342170942297e+79*XCR*(XCR + XNB - 1) - 1.1492049947789595e+79*XCR - 4.1759285975607422e+78*XNB + 1.3853995416615124e+82*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 2.3735768934135431e+77*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 7.3198819395753102e+72*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347) + 7.5019529493423158e+87*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 21) + 2.2033617604446248e+83*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 11) + 4.1759285975673398e+78)
	                                                                                                                                )
	                                                                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                        (1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3)*(9.7869720868075301e+76*XCR*(XCR + XNB - 1) - 4.7292386616418077e+76*XCR - 1.7184891347986593e+76*XNB + 1.6350975644205316e+70*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.654186547737425e+70*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 3.0122970944754354e+70*(3605*XCR - 633)*(7210*XCR + 3605*XNB - 5347) + 8.8540703436034745e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 21) + 1.5355606949328051e+76*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 11) + 1.7184891347905147e+76)
	                                                                                                                                        )
	                                                                                                                                        : (
	                                                                                                                                                0
	                                                                                                                                        )))))*log(-(-1.9099999999999999*XCR*(XCR + XNB - 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                -0.33333333333333298
	                                                                                                                                                )
	                                                                                                                                                : (
	                                                                                                                                                        1.0
	                                                                                                                                                )) + 1) + 25412689879.149994 + 950472067.50000012*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                        )
	                                                                                                                                                        : (
	                                                                                                                                                                1.0
	                                                                                                                                                        ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                                                                0.90437860600165965*XCR - 1.0/721.0*(8.9245894917877369*XCR - 1.5670638414151561)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16) - 1.0/3605.0*(160.55053139320452*XCR - 28.19098096307863)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 1.0/3605.0*(1624.6423602258869*XCR - 285.27007323799899)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 0.15879935023552028
	                                                                                                                                                                )
	                                                                                                                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                                                                        -2.7131358180049818*XCR + (1.0/721.0)*(6.2196998640995823e-7*XCR - 1.0921137348058351e-7)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + (1.0/3605.0)*(0.0081568120405021791*XCR - 0.0014322502140465686)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + (1.0/3605.0)*(60.171939267625604*XCR - 10.565558268074065)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 0.47639805070656127
	                                                                                                                                                                        )
	                                                                                                                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                                                                -1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0*(3605*XCR - 633)*(5.7724980902563013e+80*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 20) + 1.6954120667239596e+76*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 4)
	                                                                                                                                                                                )
	                                                                                                                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                                                                        (1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0)*(3605*XCR - 633)*(6.8129065184188816e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.1815618198124465e+69*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4)
	                                                                                                                                                                                        )
	                                                                                                                                                                                        : (
	                                                                                                                                                                                                0
	                                                                                                                                                                                        )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                                -0.33333333333333298
	                                                                                                                                                                                                )
	                                                                                                                                                                                                : (
	                                                                                                                                                                                                        1.0
	                                                                                                                                                                                                )) + 1) + 1815401648.9250002*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                                                                        )
	                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                1.0
	                                                                                                                                                                                                        ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                                                                                                                0.90437860600165965*XCR*(XCR + XNB - 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 + 0.15022120760294844/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3) + 0.0049483905499523653/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 9) + 0.00082520476114542193/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 15)
	                                                                                                                                                                                                                )
	                                                                                                                                                                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                                                                                                                        -2.7131358180049818*XCR*(XCR + XNB - 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697484e-11/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                                                                                                                -1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0*(1.1544996180512604e+80*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 5.6513735557465315e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                                                                                                                        (1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0)*(1.3625813036837763e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 3.9385393993748223e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                0
	                                                                                                                                                                                                                                        )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                                                                                                                -0.33333333333333298
	                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                        1.0
	                                                                                                                                                                                                                                                )) + 1);
	return d2GCAL_gam_dxNbCr_result;

}

__device__ double d_d2GCAL_gam_dxNbNb(double XCR, double XNB)
{

	double d2GCAL_gam_dxNbNb_result;
	d2GCAL_gam_dxNbNb_result = 4222199999.9999995*pow(XCR, 2) + 4222199999.9999995*XCR*XNB - 4222199999.9999995*XCR*(XCR + XNB - 1) - 44791752420.699997*XCR - 83684217653.399994*XNB - 3467417149.4467502*pow(XCR - 0.27225130890052357, 2)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                               0.11111111111111088
	                           )
	                           : (
	                               1.0
	                           ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                   0.90437860600165965*XCR*(XCR + XNB - 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 + 0.15022120760294844/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3) + 0.0049483905499523653/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 9) + 0.00082520476114542193/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 15)
	                               )
	                               : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                      -2.7131358180049818*XCR*(XCR + XNB - 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697484e-11/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                  )
	                                  : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                          -1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0*(1.1544996180512604e+80*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 5.6513735557465315e+75*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                     )
	                                     : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                             (1.0/137437542533732097158773697755844333797641505617639277496433258056640625.0)*(1.3625813036837763e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 3.9385393993748223e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                        )
	                                        : (
	                                                0
	                                        )))))/pow((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                -0.33333333333333298
	                                                )
	                                                : (
	                                                        1.0
	                                                )) + 1, 2) + 1900944135.0000002*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                        -0.33333333333333298
	                                                        )
	                                                        : (
	                                                                1.0
	                                                        ))*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                0.90437860600165965*XCR - 1.0/721.0*(8.9245894917877369*XCR - 1.5670638414151561)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 16) - 1.0/3605.0*(160.55053139320452*XCR - 28.19098096307863)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) - 1.0/3605.0*(1624.6423602258869*XCR - 285.27007323799899)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4) - 0.15879935023552028
	                                                                )
	                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                        -2.7131358180049818*XCR + (1.0/721.0)*(6.2196998640995823e-7*XCR - 1.0921137348058351e-7)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + (1.0/3605.0)*(0.0081568120405021791*XCR - 0.0014322502140465686)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + (1.0/3605.0)*(60.171939267625604*XCR - 10.565558268074065)/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 0.47639805070656127
	                                                                        )
	                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                -1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0*(3605*XCR - 633)*(5.7724980902563013e+80*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 20) + 1.6954120667239596e+76*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 10) + 1.8299704848938276e+72)*pow(-XCR*(XCR + XNB - 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.1755894590848821, 4)
	                                                                                )
	                                                                                : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                        (1.0/99092468166820842051475836081963764668099525550317919074928379058837890625.0)*(3605*XCR - 633)*(6.8129065184188816e+68*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.1815618198124465e+69*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 7.5307427361885884e+69)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 4)
	                                                                                        )
	                                                                                        : (
	                                                                                                0
	                                                                                        )))))/((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                -0.33333333333333298
	                                                                                                )
	                                                                                                : (
	                                                                                                        1.0
	                                                                                                )) + 1) + 950472067.50000012*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                        1.0/XNB
	                                                                                                        )
	                                                                                                        : (
	                                                                                                                0
	                                                                                                        )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                -1/(XCR + XNB - 1)
	                                                                                                                )
	                                                                                                                : (
	                                                                                                                        0
	                                                                                                                )) + 950472067.50000012*((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 < -1143.1500000000001) ? (
	                                                                                                                        (1.0/12996025.0)*(3605*XCR - 633)*(3605*XCR - 633)*(1.8026544912353808 + 0.44535514949571298/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 6) + 0.19804914267490123/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 12))/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 5)
	                                                                                                                        )
	                                                                                                                        : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 1143.1500000000001) ? (
	                                                                                                                                -1.0/12996025.0*(3605*XCR - 633)*(3605*XCR - 633)*(0.066764981156866132 + 2.2626385687939472e-5/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 6) + 1.3802385274007393e-8/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 12))/pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                )
	                                                                                                                                : ((3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 > 0 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 1143.1500000000001) ? (
	                                                                                                                                        -1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0*(3605*XCR - 633)*(3605*XCR - 633)*(1.3853995416615124e+82*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 20) + 2.3735768934135431e+77*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) + 7.3198819395753102e+72)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3)
	                                                                                                                                        )
	                                                                                                                                        : ((1201.6666666666654*XCR*(XCR + XNB - 1) - 580.66666666666606*XCR - 210.99999999999977*XNB + 210.99999999999977 > -1143.1500000000001 && 3605*XCR*(XCR + XNB - 1) - 1742*XCR - 633*XNB + 633 < 0) ? (
	                                                                                                                                                (1.0/357228347741389135595570389075479371628498789608896098265116806507110595703125.0)*(3605*XCR - 633)*(3605*XCR - 633)*(1.6350975644205316e+70*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 20) + 1.654186547737425e+70*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 10) + 3.0122970944754354e+70)*pow(XCR*(XCR + XNB - 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.17558945908377255, 3)
	                                                                                                                                                )
	                                                                                                                                                : (
	                                                                                                                                                        0
	                                                                                                                                                )))))*log(-(-1.9099999999999999*XCR*(XCR + XNB - 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002)*((1.9099999999999999*XCR*(XCR + XNB - 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002 <= 0) ? (
	                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                        )
	                                                                                                                                                        : (
	                                                                                                                                                                1.0
	                                                                                                                                                        )) + 1) + 57534684916.199982;
	return d2GCAL_gam_dxNbNb_result;

}

__device__ double d_d2GCAL_del_dxCrCr(double XCR)
{

	double d2GCAL_del_dxCrCr_result;
	d2GCAL_del_dxCrCr_result = 712854050.625*(((1.3333333333333333*XCR > 9.9999999999999998e-17) ? (
	                               1.3333333333333333/XCR
	                           )
	                           : (
	                               0
	                           )) + ((1.3333333333333333*XCR - 1 < -9.9999999999999998e-17) ? (
	                                     -1.7777777777777777/(1.3333333333333333*XCR - 1)
	                                 )
	                                 : (
	                                     0
	                                 )));
	return d2GCAL_del_dxCrCr_result;

}

__device__ double d_d2GCAL_del_dxCrNb(double XNB)
{

	double d2GCAL_del_dxCrNb_result;
	d2GCAL_del_dxCrNb_result = 20189402430.399982 - 13139733333.33333*XNB;
	return d2GCAL_del_dxCrNb_result;

}

__device__ double d_d2GCAL_del_dxNbCr(double XNB)
{

	double d2GCAL_del_dxNbCr_result;
	d2GCAL_del_dxNbCr_result = 20189402430.399982 - 13139733333.33333*XNB;
	return d2GCAL_del_dxNbCr_result;

}

__device__ double d_d2GCAL_del_dxNbNb(double XCR, double XNB)
{

	double d2GCAL_del_dxNbNb_result;
	d2GCAL_del_dxNbNb_result = -13139733333.33333*XCR + 237618016.87500003*((4*XNB > 9.9999999999999998e-17) ? (
	                               4/XNB
	                           )
	                           : (
	                               0
	                           )) + 237618016.87500003*((4*XNB - 1 < -9.9999999999999998e-17) ? (
	                                       -16/(4*XNB - 1)
	                                   )
	                                   : (
	                                       0
	                                   )) + 9854799999.9999981;
	return d2GCAL_del_dxNbNb_result;

}

__device__ double d_d2GCAL_lav_dxCrCr(double XCR, double XNB)
{

	double d2GCAL_lav_dxCrCr_result;
	d2GCAL_lav_dxCrCr_result = 46344062414.699997*XNB + 633648045.0*((1.5*(XCR + XNB - 1) < -9.9999999999999998e-17) ? (
	                               -1.5/(XCR + XNB - 1)
	                           )
	                           : (
	                               0
	                           )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 9.9999999999999998e-17) ? (
	                                       2.25/(1.5*XCR + 1.5*XNB - 0.5)
	                                   )
	                                   : (
	                                       0
	                                   ));
	return d2GCAL_lav_dxCrCr_result;

}

__device__ double d_d2GCAL_lav_dxCrNb(double XCR, double XNB)
{

	double d2GCAL_lav_dxCrNb_result;
	d2GCAL_lav_dxCrNb_result = 46344062414.699997*XCR - 344230260485.84985*XNB + 633648045.0*((1.5*(XCR + XNB - 1) < -9.9999999999999998e-17) ? (
	                               -1.5/(XCR + XNB - 1)
	                           )
	                           : (
	                               0
	                           )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 9.9999999999999998e-17) ? (
	                                       2.25/(1.5*XCR + 1.5*XNB - 0.5)
	                                   )
	                                   : (
	                                       0
	                                   )) + 89999259137.618134;
	return d2GCAL_lav_dxCrNb_result;

}

__device__ double d_d2GCAL_lav_dxNbCr(double XCR, double XNB)
{

	double d2GCAL_lav_dxNbCr_result;
	d2GCAL_lav_dxNbCr_result = 46344062414.699997*XCR - 344230260485.84985*XNB + 633648045.0*((1.5*(XCR + XNB - 1) < -9.9999999999999998e-17) ? (
	                               -1.5/(XCR + XNB - 1)
	                           )
	                           : (
	                               0
	                           )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 9.9999999999999998e-17) ? (
	                                       2.25/(1.5*XCR + 1.5*XNB - 0.5)
	                                   )
	                                   : (
	                                       0
	                                   )) + 89999259137.618134;
	return d2GCAL_lav_dxNbCr_result;

}

__device__ double d_d2GCAL_lav_dxNbNb(double XCR, double XNB)
{

	double d2GCAL_lav_dxNbNb_result;
	d2GCAL_lav_dxNbNb_result = -344230260485.84991*XCR - 1171722968701.6499*XNB + 316824022.5*((3*XNB > 9.9999999999999998e-17) ? (
	                               3/XNB
	                           )
	                           : (
	                               0
	                           )) + 316824022.5*((3*XNB - 1 < -9.9999999999999998e-17) ? (
	                                       -9/(3*XNB - 1)
	                                   )
	                                   : (
	                                       0
	                                   )) + 633648045.0*((1.5*(XCR + XNB - 1) < -9.9999999999999998e-17) ? (
	                                           -1.5/(XCR + XNB - 1)
	                                           )
	                                           : (
	                                                   0
	                                           )) + 633648045.0*((1.5*XCR + 1.5*XNB - 0.5 > 9.9999999999999998e-17) ? (
	                                                   2.25/(1.5*XCR + 1.5*XNB - 0.5)
	                                                   )
	                                                   : (
	                                                           0
	                                                   )) + 605810087590.48621;
	return d2GCAL_lav_dxNbNb_result;

}

__device__ double d_d2g_gam_dxCrCr()
{

	double d2g_gam_dxCrCr_result;
	d2g_gam_dxCrCr_result = 4534264425.5240498;
	return d2g_gam_dxCrCr_result;

}

__device__ double d_d2g_gam_dxCrNb()
{

	double d2g_gam_dxCrNb_result;
	d2g_gam_dxCrNb_result = 15095482346.486225;
	return d2g_gam_dxCrNb_result;

}

__device__ double d_d2g_gam_dxNbCr()
{

	double d2g_gam_dxNbCr_result;
	d2g_gam_dxNbCr_result = 15095482346.486225;
	return d2g_gam_dxNbCr_result;

}

__device__ double d_d2g_gam_dxNbNb()
{

	double d2g_gam_dxNbNb_result;
	d2g_gam_dxNbNb_result = 110386166481.37187;
	return d2g_gam_dxNbNb_result;

}

__device__ double d_d2g_del_dxCrCr()
{

	double d2g_del_dxCrCr_result;
	d2g_del_dxCrCr_result = 42692985981.597771;
	return d2g_del_dxCrCr_result;

}

__device__ double d_d2g_del_dxCrNb()
{

	double d2g_del_dxCrNb_result;
	d2g_del_dxCrNb_result = 16906497386.287418;
	return d2g_del_dxCrNb_result;

}

__device__ double d_d2g_del_dxNbCr()
{

	double d2g_del_dxNbCr_result;
	d2g_del_dxNbCr_result = 16906497386.287418;
	return d2g_del_dxNbCr_result;

}

__device__ double d_d2g_del_dxNbNb()
{

	double d2g_del_dxNbNb_result;
	d2g_del_dxNbNb_result = 6170738265863.7695;
	return d2g_del_dxNbNb_result;

}

__device__ double d_d2g_lav_dxCrCr()
{

	double d2g_lav_dxCrCr_result;
	d2g_lav_dxCrCr_result = 17733460569.613991;
	return d2g_lav_dxCrCr_result;

}

__device__ double d_d2g_lav_dxCrNb()
{

	double d2g_lav_dxCrNb_result;
	d2g_lav_dxCrNb_result = 24191004361.532181;
	return d2g_lav_dxCrNb_result;

}

__device__ double d_d2g_lav_dxNbCr()
{

	double d2g_lav_dxNbCr_result;
	d2g_lav_dxNbCr_result = 24191004361.532181;
	return d2g_lav_dxNbCr_result;

}

__device__ double d_d2g_lav_dxNbNb()
{

	double d2g_lav_dxNbNb_result;
	d2g_lav_dxNbNb_result = 196588620559.76782;
	return d2g_lav_dxNbNb_result;

}

__device__ double d_mu_Cr(double XCR, double XNB)
{

	double mu_Cr_result;
	mu_Cr_result = -21111.0*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21111.0*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 6693.8863150000006*XCR*(-XCR - XNB + 1) + 17073.006801792686*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 78462.880447499992*XNB*(-XCR - XNB + 1) - 1.0000000000000001e-5*XNB*(2111099999.9999998*pow(XCR, 2)*XNB - 2111099999.9999998*pow(XCR, 2)*(-XCR - XNB + 1) + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) - 2111099999.9999998*XCR*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999995*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 8515676676.2499981*XCR + 13947369608.899998*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 15692576089.499996*XNB + 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                   -0.33333333333333298
	               )
	               : (
	                   1.0
	               ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                       -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                   )
	                   : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                          2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                      )
	                      : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                             -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                         )
	                         : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                            )
	                            : (
	                                0
	                            )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                        -0.33333333333333298
	                                    )
	                                    : (
	                                        1.0
	                                    )) + 1) + (-6973684804.4499989*XCR - 6973684804.4499989*XNB + 6973684804.4499989)*(XCR + 2*XNB - 1) + 950472067.50000012*((XNB > 9.9999999999999998e-17) ? (
	                                            log(XNB) + 1
	                                            )
	                                            : (
	                                                    0
	                                            )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                    -log(-XCR - XNB + 1) - 1
	                                                    )
	                                                    : (
	                                                            0
	                                                    )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                            0.90437860600165965*XCR - 0.15022120760294841*(3*XCR - 1899.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(9*XCR - 5697.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(15*XCR - 1899.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 0.15879935023552028
	                                                            )
	                                                            : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                    -2.7131358180049818*XCR - 0.0055637484297388446*(1899.0/3605.0 - 3*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 2.5140428542154968e-7*(5697.0/3605.0 - 9*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 5.7509938641697477e-11*(1899.0/721.0 - 15*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + 0.47639805070656127
	                                                                    )
	                                                                    : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                            -13.314924373336252*(5*XCR - 633.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(15*XCR - 1899.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(25*XCR - 3165.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                                            )
	                                                                            : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                    -0.054793927462289095*(633.0/721.0 - 5*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0028656939921696892*(1899.0/721.0 - 15*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.00099141855897878122*(3165.0/721.0 - 25*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24)
	                                                                                    )
	                                                                                    : (
	                                                                                            0
	                                                                                    )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                            -0.33333333333333298
	                                                                                            )
	                                                                                            : (
	                                                                                                    1.0
	                                                                                            )) + 1) - 6777241581.9992752) + 10690.46462750723*XNB + 1.0000000000000001e-5*(1 - XCR)*(2111099999.9999998*pow(XCR, 2)*XNB + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) + 2949643593.999999*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 1338777263.0*XCR - 2111099999.9999998*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 6973684804.4499989*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 8515676676.2499981*XNB + (-1474821796.9999995*XCR - 1474821796.9999995*XNB + 1474821796.9999995)*(2*XCR + XNB - 1) + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
	                                                                                                    log(XCR) + 1
	                                                                                                    )
	                                                                                                    : (
	                                                                                                            0
	                                                                                                    )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                            -log(-XCR - XNB + 1) - 1
	                                                                                                            )
	                                                                                                            : (
	                                                                                                                    0
	                                                                                                            )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                    1.8087572120033193*XCR + 0.90437860600165965*XNB - 0.15022120760294841*(6*XCR + 3*XNB - 16041.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(18*XCR + 9*XNB - 48123.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(30*XCR + 15*XNB - 16041.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 1.3413904039641813
	                                                                                                                    )
	                                                                                                                    : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                            -5.4262716360099637*XCR - 2.7131358180049818*XNB - 5.7509938641697477e-11*(-30*XCR - 15*XNB + 16041.0/721.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) - 2.5140428542154968e-7*(-18*XCR - 9*XNB + 48123.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 0.0055637484297388446*(-6*XCR - 3*XNB + 16041.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.0241712118925488
	                                                                                                                            )
	                                                                                                                            : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                    -13.314924373336252*(10*XCR + 5*XNB - 5347.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(30*XCR + 15*XNB - 16041.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(50*XCR + 25*XNB - 26735.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                                                                                                    )
	                                                                                                                                    : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                            -0.00099141855897878122*(-50*XCR - 25*XNB + 26735.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24) - 0.0028656939921696892*(-30*XCR - 15*XNB + 16041.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.054793927462289095*(-10*XCR - 5*XNB + 5347.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4)
	                                                                                                                                            )
	                                                                                                                                            : (
	                                                                                                                                                    0
	                                                                                                                                            )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                    -0.33333333333333298
	                                                                                                                                                    )
	                                                                                                                                                    : (
	                                                                                                                                                            1.0
	                                                                                                                                                    )) + 1) + 1037912048.6792686 + 950472067.50000012*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                            -0.33333333333333298
	                                                                                                                                                            )
	                                                                                                                                                            : (
	                                                                                                                                                                    1.0
	                                                                                                                                                            ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                                    -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                                                                                    )
	                                                                                                                                                                    : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                                            2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                                            )
	                                                                                                                                                                            : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                                    -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                                    )
	                                                                                                                                                                                    : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                                            -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                                                                                            )
	                                                                                                                                                                                            : (
	                                                                                                                                                                                                    0
	                                                                                                                                                                                            )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                                    -0.33333333333333298
	                                                                                                                                                                                                    )
	                                                                                                                                                                                                    : (
	                                                                                                                                                                                                            1.0
	                                                                                                                                                                                                    )) + 1)) + 9504.7206750000023*((XCR > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                                            XCR*log(XCR)
	                                                                                                                                                                                                            )
	                                                                                                                                                                                                            : (
	                                                                                                                                                                                                                    0
	                                                                                                                                                                                                            )) + 9504.7206750000023*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                    XNB*log(XNB)
	                                                                                                                                                                                                                    )
	                                                                                                                                                                                                                    : (
	                                                                                                                                                                                                                            0
	                                                                                                                                                                                                                    )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                            (-XCR - XNB + 1)*log(-XCR - XNB + 1)
	                                                                                                                                                                                                                            )
	                                                                                                                                                                                                                            : (
	                                                                                                                                                                                                                                    0
	                                                                                                                                                                                                                            )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                                                                                                    -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                                                                                                                                                    )
	                                                                                                                                                                                                                                    : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                                                                                                            2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                                                                                                            )
	                                                                                                                                                                                                                                            : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                                                                                                    -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                                                                                                    )
	                                                                                                                                                                                                                                                    : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                                                                                                            -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                                                                                                                                                            )
	                                                                                                                                                                                                                                                            : (
	                                                                                                                                                                                                                                                                    0
	                                                                                                                                                                                                                                                            )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                                                                                                    -0.33333333333333298
	                                                                                                                                                                                                                                                                    )
	                                                                                                                                                                                                                                                                    : (
	                                                                                                                                                                                                                                                                            1.0
	                                                                                                                                                                                                                                                                    )) + 1) - 54642.776948364204;
	return mu_Cr_result;

}

__device__ double d_mu_Nb(double XCR, double XNB)
{

	double mu_Nb_result;
	mu_Nb_result = -21111.0*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21111.0*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 6693.8863150000006*XCR*(-XCR - XNB + 1) - 1.0000000000000001e-5*XCR*(2111099999.9999998*pow(XCR, 2)*XNB + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) + 2949643593.999999*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 1338777263.0*XCR - 2111099999.9999998*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 6973684804.4499989*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 8515676676.2499981*XNB + (-1474821796.9999995*XCR - 1474821796.9999995*XNB + 1474821796.9999995)*(2*XCR + XNB - 1) + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
	                   log(XCR) + 1
	               )
	               : (
	                   0
	               )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                       -log(-XCR - XNB + 1) - 1
	                                        )
	                                        : (
	                                                0
	                                        )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                1.8087572120033193*XCR + 0.90437860600165965*XNB - 0.15022120760294841*(6*XCR + 3*XNB - 16041.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(18*XCR + 9*XNB - 48123.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(30*XCR + 15*XNB - 16041.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 1.3413904039641813
	                                                )
	                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                        -5.4262716360099637*XCR - 2.7131358180049818*XNB - 5.7509938641697477e-11*(-30*XCR - 15*XNB + 16041.0/721.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) - 2.5140428542154968e-7*(-18*XCR - 9*XNB + 48123.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 0.0055637484297388446*(-6*XCR - 3*XNB + 16041.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.0241712118925488
	                                                        )
	                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                -13.314924373336252*(10*XCR + 5*XNB - 5347.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(30*XCR + 15*XNB - 16041.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(50*XCR + 25*XNB - 26735.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                                )
	                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                        -0.00099141855897878122*(-50*XCR - 25*XNB + 26735.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24) - 0.0028656939921696892*(-30*XCR - 15*XNB + 16041.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.054793927462289095*(-10*XCR - 5*XNB + 5347.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4)
	                                                                        )
	                                                                        : (
	                                                                                0
	                                                                        )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                -0.33333333333333298
	                                                                                )
	                                                                                : (
	                                                                                        1.0
	                                                                                )) + 1) + 1037912048.6792686 + 950472067.50000012*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                        -0.33333333333333298
	                                                                                        )
	                                                                                        : (
	                                                                                                1.0
	                                                                                        ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                )
	                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                        2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                        )
	                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                )
	                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                        -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                        )
	                                                                                                                        : (
	                                                                                                                                0
	                                                                                                                        )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                -0.33333333333333298
	                                                                                                                                )
	                                                                                                                                : (
	                                                                                                                                        1.0
	                                                                                                                                )) + 1)) + 17073.006801792686*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 78462.880447499992*XNB*(-XCR - XNB + 1) + 10690.46462750723*XNB + 1.0000000000000001e-5*(1 - XNB)*(2111099999.9999998*pow(XCR, 2)*XNB - 2111099999.9999998*pow(XCR, 2)*(-XCR - XNB + 1) + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) - 2111099999.9999998*XCR*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999995*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 8515676676.2499981*XCR + 13947369608.899998*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 15692576089.499996*XNB + 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                        -0.33333333333333298
	                                                                                                                                        )
	                                                                                                                                        : (
	                                                                                                                                                1.0
	                                                                                                                                        ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                                                                )
	                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                        2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                        )
	                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                )
	                                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                        -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                                                                        )
	                                                                                                                                                                        : (
	                                                                                                                                                                                0
	                                                                                                                                                                        )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                -0.33333333333333298
	                                                                                                                                                                                )
	                                                                                                                                                                                : (
	                                                                                                                                                                                        1.0
	                                                                                                                                                                                )) + 1) + (-6973684804.4499989*XCR - 6973684804.4499989*XNB + 6973684804.4499989)*(XCR + 2*XNB - 1) + 950472067.50000012*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                        log(XNB) + 1
	                                                                                                                                                                                        )
	                                                                                                                                                                                        : (
	                                                                                                                                                                                                0
	                                                                                                                                                                                        )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                                                                                                -log(-XCR - XNB + 1) - 1
	                                                                                                                                                                                                )
	                                                                                                                                                                                                : (
	                                                                                                                                                                                                        0
	                                                                                                                                                                                                )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                                                                        0.90437860600165965*XCR - 0.15022120760294841*(3*XCR - 1899.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(9*XCR - 5697.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(15*XCR - 1899.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 0.15879935023552028
	                                                                                                                                                                                                        )
	                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                                                                                -2.7131358180049818*XCR - 0.0055637484297388446*(1899.0/3605.0 - 3*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 2.5140428542154968e-7*(5697.0/3605.0 - 9*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 5.7509938641697477e-11*(1899.0/721.0 - 15*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + 0.47639805070656127
	                                                                                                                                                                                                                )
	                                                                                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                                                                        -13.314924373336252*(5*XCR - 633.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(15*XCR - 1899.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(25*XCR - 3165.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                                                                                -0.054793927462289095*(633.0/721.0 - 5*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0028656939921696892*(1899.0/721.0 - 15*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.00099141855897878122*(3165.0/721.0 - 25*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24)
	                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                1.0
	                                                                                                                                                                                                                                        )) + 1) - 6777241581.9992752) + 9504.7206750000023*((XCR > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                                                XCR*log(XCR)
	                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                                )) + 9504.7206750000023*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                                                        XNB*log(XNB)
	                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                                0
	                                                                                                                                                                                                                                                        )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                                                                (-XCR - XNB + 1)*log(-XCR - XNB + 1)
	                                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                                                )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                                                                                                                                        -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                                                                                                                                                2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                                                                                                                                        -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                                                                                                                                                -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                                                                                )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                                                                                1.0
	                                                                                                                                                                                                                                                                                                        )) + 1) - 54642.776948364204;
	return mu_Nb_result;

}

__device__ double d_mu_Ni(double XCR, double XNB)
{

	double mu_Ni_result;
	mu_Ni_result = -21111.0*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 21111.0*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 14748.217969999996*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 6693.8863150000006*XCR*(-XCR - XNB + 1) - 1.0000000000000001e-5*XCR*(2111099999.9999998*pow(XCR, 2)*XNB + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) + 2949643593.999999*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 1338777263.0*XCR - 2111099999.9999998*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 6973684804.4499989*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 8515676676.2499981*XNB + (-1474821796.9999995*XCR - 1474821796.9999995*XNB + 1474821796.9999995)*(2*XCR + XNB - 1) + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
	                   log(XCR) + 1
	               )
	               : (
	                   0
	               )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                       -log(-XCR - XNB + 1) - 1
	                                        )
	                                        : (
	                                                0
	                                        )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                1.8087572120033193*XCR + 0.90437860600165965*XNB - 0.15022120760294841*(6*XCR + 3*XNB - 16041.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(18*XCR + 9*XNB - 48123.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(30*XCR + 15*XNB - 16041.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 1.3413904039641813
	                                                )
	                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                        -5.4262716360099637*XCR - 2.7131358180049818*XNB - 5.7509938641697477e-11*(-30*XCR - 15*XNB + 16041.0/721.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) - 2.5140428542154968e-7*(-18*XCR - 9*XNB + 48123.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 0.0055637484297388446*(-6*XCR - 3*XNB + 16041.0/3605.0)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) + 4.0241712118925488
	                                                        )
	                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                -13.314924373336252*(10*XCR + 5*XNB - 5347.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(30*XCR + 15*XNB - 16041.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(50*XCR + 25*XNB - 26735.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                                )
	                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                        -0.00099141855897878122*(-50*XCR - 25*XNB + 26735.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24) - 0.0028656939921696892*(-30*XCR - 15*XNB + 16041.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.054793927462289095*(-10*XCR - 5*XNB + 5347.0/721.0)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4)
	                                                                        )
	                                                                        : (
	                                                                                0
	                                                                        )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                -0.33333333333333298
	                                                                                )
	                                                                                : (
	                                                                                        1.0
	                                                                                )) + 1) + 1037912048.6792686 + 950472067.50000012*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                        -0.33333333333333298
	                                                                                        )
	                                                                                        : (
	                                                                                                1.0
	                                                                                        ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                )
	                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                        2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                        )
	                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                )
	                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                        -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                        )
	                                                                                                                        : (
	                                                                                                                                0
	                                                                                                                        )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                -0.33333333333333298
	                                                                                                                                )
	                                                                                                                                : (
	                                                                                                                                        1.0
	                                                                                                                                )) + 1)) + 17073.006801792686*XCR + 69736.848044499988*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 78462.880447499992*XNB*(-XCR - XNB + 1) - 1.0000000000000001e-5*XNB*(2111099999.9999998*pow(XCR, 2)*XNB - 2111099999.9999998*pow(XCR, 2)*(-XCR - XNB + 1) + 2111099999.9999998*XCR*pow(XNB, 2) - 4222199999.9999995*XCR*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(2*XCR + 2*XNB - 2) - 2111099999.9999998*XCR*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999995*XCR*(-XCR - XNB + 1) - 1474821796.9999995*XCR*(2*XCR + XNB - 1) + 8515676676.2499981*XCR + 13947369608.899998*XNB*(-XCR - XNB + 1) - 6973684804.4499989*XNB*(XCR + 2*XNB - 1) + 15692576089.499996*XNB + 950472067.50000012*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                        -0.33333333333333298
	                                                                                                                                        )
	                                                                                                                                        : (
	                                                                                                                                                1.0
	                                                                                                                                        ))*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                                                                )
	                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                        2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                        )
	                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                )
	                                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                        -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                                                                        )
	                                                                                                                                                                        : (
	                                                                                                                                                                                0
	                                                                                                                                                                        )))))/((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                -0.33333333333333298
	                                                                                                                                                                                )
	                                                                                                                                                                                : (
	                                                                                                                                                                                        1.0
	                                                                                                                                                                                )) + 1) + (-6973684804.4499989*XCR - 6973684804.4499989*XNB + 6973684804.4499989)*(XCR + 2*XNB - 1) + 950472067.50000012*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                        log(XNB) + 1
	                                                                                                                                                                                        )
	                                                                                                                                                                                        : (
	                                                                                                                                                                                                0
	                                                                                                                                                                                        )) + 950472067.50000012*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                                                                                                -log(-XCR - XNB + 1) - 1
	                                                                                                                                                                                                )
	                                                                                                                                                                                                : (
	                                                                                                                                                                                                        0
	                                                                                                                                                                                                )) + 950472067.50000012*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                                                                        0.90437860600165965*XCR - 0.15022120760294841*(3*XCR - 1899.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0049483905499523662*(9*XCR - 5697.0/3605.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 10) - 0.00082520476114542182*(15*XCR - 1899.0/721.0)/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 16) - 0.15879935023552028
	                                                                                                                                                                                                        )
	                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                                                                                -2.7131358180049818*XCR - 0.0055637484297388446*(1899.0/3605.0 - 3*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 2.5140428542154968e-7*(5697.0/3605.0 - 9*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 10) - 5.7509938641697477e-11*(1899.0/721.0 - 15*XCR)/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 16) + 0.47639805070656127
	                                                                                                                                                                                                                )
	                                                                                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                                                                        -13.314924373336252*(5*XCR - 633.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 4) - 41119.576584101698*(15*XCR - 1899.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 14) - 840017652.21311688*(25*XCR - 3165.0/721.0)*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 24)
	                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                                                                                -0.054793927462289095*(633.0/721.0 - 5*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 4) - 0.0028656939921696892*(1899.0/721.0 - 15*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 14) - 0.00099141855897878122*(3165.0/721.0 - 25*XCR)*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 24)
	                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                1.0
	                                                                                                                                                                                                                                        )) + 1) - 6777241581.9992752) + 10690.46462750723*XNB + 9504.7206750000023*((XCR > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                                                XCR*log(XCR)
	                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                                )) + 9504.7206750000023*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                                                        XNB*log(XNB)
	                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                                0
	                                                                                                                                                                                                                                                        )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                                                                                                                                                                (-XCR - XNB + 1)*log(-XCR - XNB + 1)
	                                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                                                )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
	                                                                                                                                                                                                                                                                        -0.90437860600165965*XCR*(-XCR - XNB + 1) - 0.43701179796252176*XCR - 0.15879935023552028*XNB + 1.1587993502347678 - 0.15022120760294841/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 3) - 0.0049483905499523662/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 9) - 0.00082520476114542182/pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15)
	                                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < -1143.1500000000001) ? (
	                                                                                                                                                                                                                                                                                2.7131358180049818*XCR*(-XCR - XNB + 1) + 1.3110353938875667*XCR + 0.47639805070656127*XNB + 0.52360194929268611 - 0.0055637484297388446/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 3) - 2.5140428542154968e-7/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 9) - 5.7509938641697477e-11/pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15)
	                                                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                                                : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > -1143.1500000000001 && 3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 < 0) ? (
	                                                                                                                                                                                                                                                                                        -840017652.21311688*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 25) - 41119.576584101698*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 15) - 13.314924373336252*pow(-XCR*(-XCR - XNB + 1) - 1742.0/3605.0*XCR - 633.0/3605.0*XNB + 0.1755894590848821, 5)
	                                                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                                                        : ((3605*XCR*(-XCR - XNB + 1) + 1742*XCR + 633*XNB - 633 > 0 && 1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 < 1143.1500000000001) ? (
	                                                                                                                                                                                                                                                                                                -0.00099141855897878122*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 25) - 0.0028656939921696892*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 15) - 0.054793927462289095*pow(XCR*(-XCR - XNB + 1) + (1742.0/3605.0)*XCR + (633.0/3605.0)*XNB - 0.17558945908377255, 5)
	                                                                                                                                                                                                                                                                                                )
	                                                                                                                                                                                                                                                                                                : (
	                                                                                                                                                                                                                                                                                                        0
	                                                                                                                                                                                                                                                                                                )))))*log((-1.9099999999999999*XCR*(-XCR - XNB + 1) - 2.98*XCR - 0.52000000000000002*XNB + 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
	                                                                                                                                                                                                                                                                                                        -0.33333333333333298
	                                                                                                                                                                                                                                                                                                        )
	                                                                                                                                                                                                                                                                                                        : (
	                                                                                                                                                                                                                                                                                                                1.0
	                                                                                                                                                                                                                                                                                                        )) + 1) - 54642.776948364204;
	return mu_Ni_result;

}

__device__ double d_D_CrCr(double XCR, double XNB)
{

	double D_CrCr_result;
	D_CrCr_result = -1.4170990277916469e-20*XCR*XNB*(-45342.644255240499*XCR - 150954.82346486227*XNB + 150954.82346486227)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 3.5027570743586952e-21*XCR*(1 - XCR)*(-45342.644255240499*XCR - 150954.82346486227*XNB + 45342.644255240499)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.8295676400933012e-21*XCR*(-45342.644255240499*XCR - 150954.82346486227*XNB)*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_CrCr_result;

}

__device__ double d_D_CrNb(double XCR, double XNB)
{

	double D_CrNb_result;
	D_CrNb_result = -1.4170990277916469e-20*XCR*XNB*(-150954.82346486227*XCR - 1103861.6648137188*XNB + 1103861.664813719)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 3.5027570743586952e-21*XCR*(1 - XCR)*(-150954.82346486227*XCR - 1103861.6648137188*XNB + 150954.82346486225)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) - 1.8295676400933012e-21*XCR*(-150954.82346486227*XCR - 1103861.6648137188*XNB)*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_CrNb_result;

}

__device__ double d_D_NbCr(double XCR, double XNB)
{

	double D_NbCr_result;
	D_NbCr_result = -3.5027570743586952e-21*XCR*XNB*(-45342.644255240499*XCR - 150954.82346486227*XNB + 45342.644255240499)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.4170990277916469e-20*XNB*(1 - XNB)*(-45342.644255240499*XCR - 150954.82346486227*XNB + 150954.82346486227)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.8295676400933012e-21*XNB*(-45342.644255240499*XCR - 150954.82346486227*XNB)*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_NbCr_result;

}

__device__ double d_D_NbNb(double XCR, double XNB)
{

	double D_NbNb_result;
	D_NbNb_result = -3.5027570743586952e-21*XCR*XNB*(-150954.82346486227*XCR - 1103861.6648137188*XNB + 150954.82346486225)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.4170990277916469e-20*XNB*(1 - XNB)*(-150954.82346486227*XCR - 1103861.6648137188*XNB + 1103861.664813719)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) - 1.8295676400933012e-21*XNB*(-150954.82346486227*XCR - 1103861.6648137188*XNB)*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_NbNb_result;

}
