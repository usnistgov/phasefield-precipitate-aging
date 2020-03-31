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

__device__ double d_inv_fict_det(double pDel, double pGam, double pLav)
{

	double inv_fict_det_result;
	inv_fict_det_result = 1.0479034445859148/(0.001036042775998007*pow(pDel, 2) + 0.12229015747171916*pDel*pGam + 0.041785936156220206*pDel*pLav + 1.0*pow(pGam, 2) + 0.73028945399293654*pGam*pLav + 0.093983883345629848*pow(pLav, 2));
	return inv_fict_det_result;

}

__device__ double d_fict_gam_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_gam_Cr_result;
	fict_gam_Cr_result = 0.95428639457822939*INV_DET*(0.016938286205500769*XCR*pDel + 1.0*XCR*pGam + 0.54889980006097583*XCR*pLav - 0.34687466604954664*XNB*pDel - 0.10246420218061991*XNB*pLav + 0.086819224054883931*pow(pDel, 2) + 0.036767802923980242*pDel*pGam + 0.11815012520929236*pDel*pLav - 0.27750258277182049*pGam*pLav - 0.12951476250779617*pow(pLav, 2));
	return fict_gam_Cr_result;

}

__device__ double d_fict_gam_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_gam_Nb_result;
	fict_gam_Nb_result = -0.95428639457822939*INV_DET*(0.0021576593653206372*XCR*pDel + 0.054466450838266962*XCR*pLav - 0.10535187126621839*XNB*pDel - 1.0*XNB*pGam - 0.18138965393196071*XNB*pLav + 0.026258691080522339*pow(pDel, 2) + 0.24849448207756411*pDel*pGam + 0.06992726713680375*pDel*pLav + 0.22257870755542866*pGam*pLav + 0.025258893954068835*pow(pLav, 2));
	return fict_gam_Nb_result;

}

__device__ double d_fict_del_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_del_Cr_result;
	fict_del_Cr_result = 0.33101777443494923*INV_DET*(0.0029867928603642142*XCR*pDel + 0.30371739875396442*XCR*pGam + 0.11224396861282378*XCR*pLav + 1.0*XNB*pGam + 0.15026949298026257*XNB*pLav - 0.25028989589710454*pDel*pGam - 0.038472217853981416*pDel*pLav - 0.10599737173866837*pow(pGam, 2) - 0.38426983287411115*pGam*pLav - 0.074556825343604541*pow(pLav, 2));
	return fict_del_Cr_result;

}

__device__ double d_fict_del_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_del_Nb_result;
	fict_del_Nb_result = 0.23713490337438312*INV_DET*(0.0086829266681549624*XCR*pGam + 0.0010534256500958393*XCR*pLav + 0.0041692787998190662*XNB*pDel + 0.068163631095090937*XNB*pGam + 0.011474488301439057*XNB*pLav + 0.10567112340275649*pDel*pGam + 0.038045328588644858*pDel*pLav + 1.0*pow(pGam, 2) + 0.71270814650507486*pGam*pLav + 0.091137578230976013*pow(pLav, 2));
	return fict_del_Nb_result;

}

__device__ double d_fict_lav_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_lav_Cr_result;
	fict_lav_Cr_result = 0.26481693919946725*INV_DET*(0.010275028791049912*XCR*pDel + 0.65365032685519309*XCR*pGam + 0.33867750853659306*XCR*pLav - 0.18783493715380431*XNB*pDel + 0.36923693162478494*XNB*pGam + 0.048089778433731953*pow(pDel, 2) + 0.054570103782716275*pDel*pGam + 0.093195074562756067*pDel*pLav + 1.0*pow(pGam, 2) + 0.46671552103819908*pGam*pLav);
	return fict_lav_Cr_result;

}

__device__ double d_fict_lav_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav)
{

	double fict_lav_Nb_result;
	fict_lav_Nb_result = -0.5238076111848996*INV_DET*(0.00047690026722310252*XCR*pDel - 0.09922840349407383*XCR*pGam - 0.070932051941581878*XNB*pDel - 1.0*XNB*pGam - 0.1712222947342838*XNB*pLav + 0.017223643043877631*pow(pDel, 2) + 0.19525744879034307*pDel*pGam + 0.04125923401282721*pDel*pLav - 0.40549970601319762*pow(pGam, 2) - 0.046017313089315927*pGam*pLav);
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

__device__ double d_CALPHAD_gam(double XCR, double XNB)
{

	double CALPHAD_gam_result;
	CALPHAD_gam_result = -2111099999.9999998*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999995*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 669388631.5*XCR*(-XCR - XNB + 1) + 1707300680.1792686*XCR + 6973684804.4499989*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 7846288044.7499981*XNB*(-XCR - XNB + 1) + 1069046462.7507229*XNB + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
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
	return CALPHAD_gam_result;

}

__device__ double d_CALPHAD_del(double XCR, double XNB)
{

	double CALPHAD_del_result;
	CALPHAD_del_result = -21668797081.409531*XCR*XNB - 5258769591.26929*XCR*(1 - 4*XNB) - 1231849999.9999998*XNB*(1 - 1.3333333333333333*XCR)*(1 - 4*XNB) - 34242601639.394947*XNB*(1 - 1.3333333333333333*XCR) - 4964277694.8364201*(1 - 1.3333333333333333*XCR)*(1 - 4*XNB) + 712854050.625*((1.3333333333333333*XCR > 9.9999999999999998e-17) ? (
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
	return CALPHAD_del_result;

}

__device__ double d_CALPHAD_lav(double XCR, double XNB)
{

	double CALPHAD_lav_result;
	CALPHAD_lav_result = -46695351257.249992*XNB*(1 - 3*XNB)*(-1.5*XCR - 1.5*XNB + 1.5) + 1851135999.9999998*XNB*(1 - 3*XNB)*(1.5*XCR + 1.5*XNB - 0.5) - 10298680536.599998*XNB*(-1.5*XCR - 1.5*XNB + 1.5)*(1.5*XCR + 1.5*XNB - 0.5) - 22164936866.458534*XNB*(-1.5*XCR - 1.5*XNB + 1.5) - 16694022288.40456*XNB*(1.5*XCR + 1.5*XNB - 0.5) + 4855811416.8900013*(1 - 3*XNB)*(-1.5*XCR - 1.5*XNB + 1.5) - 4004010359.6571507*(1 - 3*XNB)*(1.5*XCR + 1.5*XNB - 0.5) + 316824022.5*((3*XNB > 9.9999999999999998e-17) ? (
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
	return CALPHAD_lav_result;

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
	mu_Cr_result = -2111099999.9999998*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999993*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 669388631.5*XCR*(-XCR - XNB + 1) + 1707300680.1792684*XCR + 6973684804.4499979*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 7846288044.7499981*XNB*(-XCR - XNB + 1) - 99999.999999999985*XNB*(21111.0*pow(XCR, 2)*XNB - 21111.0*pow(XCR, 2)*(-XCR - XNB + 1) + 21111.0*XCR*pow(XNB, 2) - 42222.0*XCR*XNB*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(2*XCR + 2*XNB - 2) - 21111.0*XCR*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 85156.766762499989*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 156925.76089499998*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
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
	                                    )) + 1) + (-69736.848044499988*XCR - 69736.848044499988*XNB + 69736.848044499988)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 9.9999999999999998e-17) ? (
	                                            log(XNB) + 1
	                                            )
	                                            : (
	                                                    0
	                                            )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                    -log(-XCR - XNB + 1) - 1
	                                                    )
	                                                    : (
	                                                            0
	                                                    )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
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
	                                                                                            )) + 1) - 67772.415819992762) + 1069046462.7507229*XNB + 99999.999999999985*(1 - XCR)*(21111.0*pow(XCR, 2)*XNB + 21111.0*XCR*pow(XNB, 2) - 42222.0*XCR*XNB*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(2*XCR + 2*XNB - 2) + 29496.435939999992*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 13387.772630000001*XCR - 21111.0*pow(XNB, 2)*(-XCR - XNB + 1) - 21111.0*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 85156.766762499989*XNB + (-14748.217969999996*XCR - 14748.217969999996*XNB + 14748.217969999996)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 9.9999999999999998e-17) ? (
	                                                                                                    log(XCR) + 1
	                                                                                                    )
	                                                                                                    : (
	                                                                                                            0
	                                                                                                    )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                            -log(-XCR - XNB + 1) - 1
	                                                                                                            )
	                                                                                                            : (
	                                                                                                                    0
	                                                                                                            )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
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
	                                                                                                                                                    )) + 1) + 10379.120486792686 + 9504.7206750000023*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
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
	                                                                                                                                                                                                    )) + 1)) + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
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
	return mu_Cr_result;

}

__device__ double d_mu_Nb(double XCR, double XNB)
{

	double mu_Nb_result;
	mu_Nb_result = -2111099999.9999998*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999993*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 669388631.5*XCR*(-XCR - XNB + 1) - 99999.999999999985*XCR*(21111.0*pow(XCR, 2)*XNB + 21111.0*XCR*pow(XNB, 2) - 42222.0*XCR*XNB*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(2*XCR + 2*XNB - 2) + 29496.435939999992*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 13387.772630000001*XCR - 21111.0*pow(XNB, 2)*(-XCR - XNB + 1) - 21111.0*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 85156.766762499989*XNB + (-14748.217969999996*XCR - 14748.217969999996*XNB + 14748.217969999996)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 9.9999999999999998e-17) ? (
	                   log(XCR) + 1
	               )
	               : (
	                   0
	               )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                       -log(-XCR - XNB + 1) - 1
	                                        )
	                                        : (
	                                                0
	                                        )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
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
	                                                                                )) + 1) + 10379.120486792686 + 9504.7206750000023*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
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
	                                                                                                                                )) + 1)) + 1707300680.1792684*XCR + 6973684804.4499979*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 7846288044.7499981*XNB*(-XCR - XNB + 1) + 1069046462.7507229*XNB + 99999.999999999985*(1 - XNB)*(21111.0*pow(XCR, 2)*XNB - 21111.0*pow(XCR, 2)*(-XCR - XNB + 1) + 21111.0*XCR*pow(XNB, 2) - 42222.0*XCR*XNB*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(2*XCR + 2*XNB - 2) - 21111.0*XCR*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 85156.766762499989*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 156925.76089499998*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
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
	                                                                                                                                                                                )) + 1) + (-69736.848044499988*XCR - 69736.848044499988*XNB + 69736.848044499988)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                        log(XNB) + 1
	                                                                                                                                                                                        )
	                                                                                                                                                                                        : (
	                                                                                                                                                                                                0
	                                                                                                                                                                                        )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                                                                                                -log(-XCR - XNB + 1) - 1
	                                                                                                                                                                                                )
	                                                                                                                                                                                                : (
	                                                                                                                                                                                                        0
	                                                                                                                                                                                                )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
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
	                                                                                                                                                                                                                                        )) + 1) - 67772.415819992762) + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
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
	return mu_Nb_result;

}

__device__ double d_mu_Ni(double XCR, double XNB)
{

	double mu_Ni_result;
	mu_Ni_result = -2111099999.9999998*pow(XCR, 2)*XNB*(-XCR - XNB + 1) - 2111099999.9999998*XCR*pow(XNB, 2)*(-XCR - XNB + 1) - 2111099999.9999998*XCR*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 1474821796.9999993*XCR*(-XCR - XNB + 1)*(2*XCR + XNB - 1) - 669388631.5*XCR*(-XCR - XNB + 1) - 99999.999999999985*XCR*(21111.0*pow(XCR, 2)*XNB + 21111.0*XCR*pow(XNB, 2) - 42222.0*XCR*XNB*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(2*XCR + 2*XNB - 2) + 29496.435939999992*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 13387.772630000001*XCR - 21111.0*pow(XNB, 2)*(-XCR - XNB + 1) - 21111.0*XNB*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 69736.848044499988*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 85156.766762499989*XNB + (-14748.217969999996*XCR - 14748.217969999996*XNB + 14748.217969999996)*(2*XCR + XNB - 1) + 9504.7206750000023*((XCR > 9.9999999999999998e-17) ? (
	                   log(XCR) + 1
	               )
	               : (
	                   0
	               )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                       -log(-XCR - XNB + 1) - 1
	                                        )
	                                        : (
	                                                0
	                                        )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
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
	                                                                                )) + 1) + 10379.120486792686 + 9504.7206750000023*(3.8199999999999998*XCR + 1.9099999999999999*XNB - 4.8899999999999997)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
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
	                                                                                                                                )) + 1)) + 1707300680.1792684*XCR + 6973684804.4499979*XNB*(-XCR - XNB + 1)*(XCR + 2*XNB - 1) - 7846288044.7499981*XNB*(-XCR - XNB + 1) - 99999.999999999985*XNB*(21111.0*pow(XCR, 2)*XNB - 21111.0*pow(XCR, 2)*(-XCR - XNB + 1) + 21111.0*XCR*pow(XNB, 2) - 42222.0*XCR*XNB*(-XCR - XNB + 1) - 21111.0*XCR*XNB*(2*XCR + 2*XNB - 2) - 21111.0*XCR*(-XCR - XNB + 1)*(-XCR - XNB + 1) + 14748.217969999996*XCR*(-XCR - XNB + 1) - 14748.217969999996*XCR*(2*XCR + XNB - 1) + 85156.766762499989*XCR + 139473.69608899998*XNB*(-XCR - XNB + 1) - 69736.848044499988*XNB*(XCR + 2*XNB - 1) + 156925.76089499998*XNB + 9504.7206750000023*(1.9099999999999999*XCR - 0.52000000000000002)*((1.9099999999999999*XCR*(-XCR - XNB + 1) + 2.98*XCR + 0.52000000000000002*XNB - 0.52000000000000002 >= 0) ? (
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
	                                                                                                                                                                                )) + 1) + (-69736.848044499988*XCR - 69736.848044499988*XNB + 69736.848044499988)*(XCR + 2*XNB - 1) + 9504.7206750000023*((XNB > 9.9999999999999998e-17) ? (
	                                                                                                                                                                                        log(XNB) + 1
	                                                                                                                                                                                        )
	                                                                                                                                                                                        : (
	                                                                                                                                                                                                0
	                                                                                                                                                                                        )) + 9504.7206750000023*((XCR + XNB - 1 < -9.9999999999999998e-17) ? (
	                                                                                                                                                                                                -log(-XCR - XNB + 1) - 1
	                                                                                                                                                                                                )
	                                                                                                                                                                                                : (
	                                                                                                                                                                                                        0
	                                                                                                                                                                                                )) + 9504.7206750000023*((1201.6666666666654*XCR*(-XCR - XNB + 1) + 580.66666666666606*XCR + 210.99999999999977*XNB - 210.99999999999977 > 1143.1500000000001) ? (
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
	                                                                                                                                                                                                                                        )) + 1) - 67772.415819992762) + 1069046462.7507229*XNB + 950472067.50000012*((XCR > 9.9999999999999998e-17) ? (
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
	return mu_Ni_result;

}

__device__ double d_D_CrCr(double XCR, double XNB)
{

	double D_CrCr_result;
	D_CrCr_result = -2.1579822888347085e-10*XCR*XNB*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 2.0959745359354141e-11*XCR*(1 - XCR)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 5.6493251883169901e-12*XCR*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_CrCr_result;

}

__device__ double d_D_CrNb(double XCR, double XNB)
{

	double D_CrNb_result;
	D_CrNb_result = -1.208281722736266e-9*XCR*XNB*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 3.8421543878596279e-11*XCR*(1 - XCR)*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.3441865859458261e-11*XCR*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_CrNb_result;

}

__device__ double d_D_NbCr(double XCR, double XNB)
{

	double D_NbCr_result;
	D_NbCr_result = -2.0959745359354141e-11*XCR*XNB*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 2.1579822888347085e-10*XNB*(1 - XNB)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 5.6493251883169901e-12*XNB*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_NbCr_result;

}

__device__ double d_D_NbNb(double XCR, double XNB)
{

	double D_NbNb_result;
	D_NbNb_result = -3.8421543878596279e-11*XCR*XNB*exp(-7.1543718172640256*XCR*(-XCR - XNB + 1) + 3.3541967644499233*XCR) + 1.208281722736266e-9*XNB*(1 - XNB)*exp(-34.982563536715503*XNB*(-XCR - XNB + 1) + 14.437947842988088*XNB) + 1.3441865859458261e-11*XNB*(-XCR - XNB + 1)*exp(-8.5221193705645018*XCR*(-XCR - XNB + 1) + 4.0036674816806448*XCR - 442.62877981749585*XNB*(-XCR - XNB + 1) - 0.82164267100913768*XNB);
	return D_NbNb_result;

}
