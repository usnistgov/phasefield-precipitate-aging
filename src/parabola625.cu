/******************************************************************************
 *                       Code generated with sympy 1.4                        *
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

__device__ double d_xr_gam_Cr(double P_del, double P_lav)
{

	double xr_gam_Cr_result;
	xr_gam_Cr_result = -1.0416420522061151e-9*P_del + 7.8015873139432392e-10*P_lav + 0.52421634830562147;
	return xr_gam_Cr_result;

}

__device__ double d_xr_gam_Nb(double P_del, double P_lav)
{

	double xr_gam_Nb_result;
	xr_gam_Nb_result = 1.2685791187413322e-10*P_del - 5.4699345802775539e-11*P_lav + 0.01299272922003303;
	return xr_gam_Nb_result;

}

__device__ double d_xr_del_Cr(double P_del, double P_lav)
{

	double xr_del_Cr_result;
	xr_del_Cr_result = -6.5735143572415373e-11*P_del + 6.321736492860471e-11*P_lav + 0.022966218927631978;
	return xr_del_Cr_result;

}

__device__ double d_xr_del_Nb(double P_del, double P_lav)
{

	double xr_del_Nb_result;
	xr_del_Nb_result = -9.8753110498082084e-14*P_del + 7.568036187685703e-13*P_lav + 0.24984563695705883;
	return xr_del_Nb_result;

}

__device__ double d_xr_lav_Cr(double P_del, double P_lav)
{

	double xr_lav_Cr_result;
	xr_lav_Cr_result = -1.7594469664015904e-10*P_del + 1.3590799747212884e-10*P_lav + 0.37392129441013022;
	return xr_lav_Cr_result;

}

__device__ double d_xr_lav_Nb(double P_del, double P_lav)
{

	double xr_lav_Nb_result;
	xr_lav_Nb_result = 1.2897736836308307e-11*P_del + 1.2468017214924273e-11*P_lav + 0.25826261799015571;
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

__device__ double d_M_CrCr(double XCR, double XNB)
{

	double M_CrCr_result;
	M_CrCr_result = 1.0000000000000003e-15*pow(XCR, 2)*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB) + 1.0000000000000003e-15*pow(XCR, 2)*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) + 1.0000000000000003e-15*(1 - XCR)*(1 - XCR)*(8.2216387898155807e-8*XCR*(-XCR - XNB + 1) + 9.6755272489124536e-20*XCR + 3.5027570743586952e-21);
	return M_CrCr_result;

}

__device__ double d_M_CrNb(double XCR, double XNB)
{

	double M_CrNb_result;
	M_CrNb_result = 1.0000000000000003e-15*XCR*XNB*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) - 1.0000000000000003e-15*XCR*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*(1 - XNB) + 1.0000000000000003e-15*XNB*(1 - XCR)*(-8.2216387898155807e-8*XCR*(-XCR - XNB + 1) - 9.6755272489124536e-20*XCR - 3.5027570743586952e-21);
	return M_CrNb_result;

}

__device__ double d_M_NbCr(double XCR, double XNB)
{

	double M_NbCr_result;
	M_NbCr_result = 1.0000000000000003e-15*XCR*XNB*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) - 1.0000000000000003e-15*XCR*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*(1 - XNB) + 1.0000000000000003e-15*XNB*(1 - XCR)*(-8.2216387898155807e-8*XCR*(-XCR - XNB + 1) - 9.6755272489124536e-20*XCR - 3.5027570743586952e-21);
	return M_NbCr_result;

}

__device__ double d_M_NbNb(double XCR, double XNB)
{

	double M_NbNb_result;
	M_NbNb_result = 1.0000000000000003e-15*pow(XNB, 2)*(8.2216387898155807e-8*XCR*(-XCR - XNB + 1) + 9.6755272489124536e-20*XCR + 3.5027570743586952e-21) + 1.0000000000000003e-15*pow(XNB, 2)*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) + 1.0000000000000003e-15*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*(1 - XNB)*(1 - XNB);
	return M_NbNb_result;

}
