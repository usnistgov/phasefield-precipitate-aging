/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/
#include "parabola625.h"
#include <math.h>

double h(double x)
{

	double h_result;
	h_result = pow(x, 3)*(6.0*pow(x, 2) - 15.0*x + 10.0);
	return h_result;

}

double hprime(double x)
{

	double hprime_result;
	hprime_result = 30.0*pow(x, 2)*pow(1.0 - x, 2);
	return hprime_result;

}

double interface_profile(double z)
{

	double interface_profile_result;
	interface_profile_result = 1.0/2.0 - 1.0/2.0*tanh(z);
	return interface_profile_result;

}

double kT()
{

	double kT_result;
	kT_result = 1.5782889043500002e-20;
	return kT_result;

}

double RT()
{

	double RT_result;
	RT_result = 9504.6840941999999;
	return RT_result;

}

double Vm()
{

	double Vm_result;
	Vm_result = 1.0000000000000001e-5;
	return Vm_result;

}

double xe_gam_Cr()
{

	double xe_gam_Cr_result;
	xe_gam_Cr_result = 0.52421634830562147;
	return xe_gam_Cr_result;

}

double xe_gam_Nb()
{

	double xe_gam_Nb_result;
	xe_gam_Nb_result = 0.01299272922003303;
	return xe_gam_Nb_result;

}

double xe_del_Cr()
{

	double xe_del_Cr_result;
	xe_del_Cr_result = 0.022966218927631978;
	return xe_del_Cr_result;

}

double xe_del_Nb()
{

	double xe_del_Nb_result;
	xe_del_Nb_result = 0.24984563695705883;
	return xe_del_Nb_result;

}

double xe_lav_Cr()
{

	double xe_lav_Cr_result;
	xe_lav_Cr_result = 0.37392129441013022;
	return xe_lav_Cr_result;

}

double xe_lav_Nb()
{

	double xe_lav_Nb_result;
	xe_lav_Nb_result = 0.25826261799015571;
	return xe_lav_Nb_result;

}

double matrix_min_Cr()
{

	double matrix_min_Cr_result;
	matrix_min_Cr_result = 0.30851614493185742;
	return matrix_min_Cr_result;

}

double matrix_max_Cr()
{

	double matrix_max_Cr_result;
	matrix_max_Cr_result = 0.36263227633734074;
	return matrix_max_Cr_result;

}

double matrix_min_Nb()
{

	double matrix_min_Nb_result;
	matrix_min_Nb_result = 0.019424801579942592;
	return matrix_min_Nb_result;

}

double matrix_max_Nb()
{

	double matrix_max_Nb_result;
	matrix_max_Nb_result = 0.025522709998118311;
	return matrix_max_Nb_result;

}

double enrich_min_Cr()
{

	double enrich_min_Cr_result;
	enrich_min_Cr_result = 0.29783560803725534;
	return enrich_min_Cr_result;

}

double enrich_max_Cr()
{

	double enrich_max_Cr_result;
	enrich_max_Cr_result = 0.35636564894177969;
	return enrich_max_Cr_result;

}

double enrich_min_Nb()
{

	double enrich_min_Nb_result;
	enrich_min_Nb_result = 0.15335241484365611;
	return enrich_min_Nb_result;

}

double enrich_max_Nb()
{

	double enrich_max_Nb_result;
	enrich_max_Nb_result = 0.15955557903581488;
	return enrich_max_Nb_result;

}

double xr_gam_Cr(double P_del, double P_lav)
{

	double xr_gam_Cr_result;
	xr_gam_Cr_result = -1.0416420522061151e-9*P_del + 7.8015873139432392e-10*P_lav + 0.52421634830562147;
	return xr_gam_Cr_result;

}

double xr_gam_Nb(double P_del, double P_lav)
{

	double xr_gam_Nb_result;
	xr_gam_Nb_result = 1.2685791187413322e-10*P_del - 5.4699345802775539e-11*P_lav + 0.01299272922003303;
	return xr_gam_Nb_result;

}

double xr_del_Cr(double P_del, double P_lav)
{

	double xr_del_Cr_result;
	xr_del_Cr_result = -6.5735143572415373e-11*P_del + 6.321736492860471e-11*P_lav + 0.022966218927631978;
	return xr_del_Cr_result;

}

double xr_del_Nb(double P_del, double P_lav)
{

	double xr_del_Nb_result;
	xr_del_Nb_result = -9.8753110498082084e-14*P_del + 7.568036187685703e-13*P_lav + 0.24984563695705883;
	return xr_del_Nb_result;

}

double xr_lav_Cr(double P_del, double P_lav)
{

	double xr_lav_Cr_result;
	xr_lav_Cr_result = -1.7594469664015904e-10*P_del + 1.3590799747212884e-10*P_lav + 0.37392129441013022;
	return xr_lav_Cr_result;

}

double xr_lav_Nb(double P_del, double P_lav)
{

	double xr_lav_Nb_result;
	xr_lav_Nb_result = 1.2897736836308307e-11*P_del + 1.2468017214924273e-11*P_lav + 0.25826261799015571;
	return xr_lav_Nb_result;

}

double inv_fict_det(double f_del, double f_gam, double f_lav)
{

	double inv_fict_det_result;
	inv_fict_det_result = 1.0479034445859148/(0.001036042775998007*pow(f_del, 2) + 0.12229015747171916*f_del*f_gam + 0.041785936156220206*f_del*f_lav + 1.0*pow(f_gam, 2) + 0.73028945399293654*f_gam*f_lav + 0.093983883345629848*pow(f_lav, 2));
	return inv_fict_det_result;

}

double fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_gam_Cr_result;
	fict_gam_Cr_result = 0.95428639457822939*INV_DET*(0.016938286205500769*XCR*f_del + 1.0*XCR*f_gam + 0.54889980006097583*XCR*f_lav - 0.34687466604954664*XNB*f_del - 0.10246420218061991*XNB*f_lav + 0.086819224054883931*pow(f_del, 2) + 0.036767802923980242*f_del*f_gam + 0.11815012520929236*f_del*f_lav - 0.27750258277182049*f_gam*f_lav - 0.12951476250779617*pow(f_lav, 2));
	return fict_gam_Cr_result;

}

double fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_gam_Nb_result;
	fict_gam_Nb_result = -0.95428639457822939*INV_DET*(0.0021576593653206372*XCR*f_del + 0.054466450838266962*XCR*f_lav - 0.10535187126621839*XNB*f_del - 1.0*XNB*f_gam - 0.18138965393196071*XNB*f_lav + 0.026258691080522339*pow(f_del, 2) + 0.24849448207756411*f_del*f_gam + 0.06992726713680375*f_del*f_lav + 0.22257870755542866*f_gam*f_lav + 0.025258893954068835*pow(f_lav, 2));
	return fict_gam_Nb_result;

}

double fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_del_Cr_result;
	fict_del_Cr_result = 0.33101777443494923*INV_DET*(0.0029867928603642142*XCR*f_del + 0.30371739875396442*XCR*f_gam + 0.11224396861282378*XCR*f_lav + 1.0*XNB*f_gam + 0.15026949298026257*XNB*f_lav - 0.25028989589710454*f_del*f_gam - 0.038472217853981416*f_del*f_lav - 0.10599737173866837*pow(f_gam, 2) - 0.38426983287411115*f_gam*f_lav - 0.074556825343604541*pow(f_lav, 2));
	return fict_del_Cr_result;

}

double fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_del_Nb_result;
	fict_del_Nb_result = 0.23713490337438312*INV_DET*(0.0086829266681549624*XCR*f_gam + 0.0010534256500958393*XCR*f_lav + 0.0041692787998190662*XNB*f_del + 0.068163631095090937*XNB*f_gam + 0.011474488301439057*XNB*f_lav + 0.10567112340275649*f_del*f_gam + 0.038045328588644858*f_del*f_lav + 1.0*pow(f_gam, 2) + 0.71270814650507486*f_gam*f_lav + 0.091137578230976013*pow(f_lav, 2));
	return fict_del_Nb_result;

}

double fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_lav_Cr_result;
	fict_lav_Cr_result = 0.26481693919946725*INV_DET*(0.010275028791049912*XCR*f_del + 0.65365032685519309*XCR*f_gam + 0.33867750853659306*XCR*f_lav - 0.18783493715380431*XNB*f_del + 0.36923693162478494*XNB*f_gam + 0.048089778433731953*pow(f_del, 2) + 0.054570103782716275*f_del*f_gam + 0.093195074562756067*f_del*f_lav + 1.0*pow(f_gam, 2) + 0.46671552103819908*f_gam*f_lav);
	return fict_lav_Cr_result;

}

double fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_lav_Nb_result;
	fict_lav_Nb_result = -0.5238076111848996*INV_DET*(0.00047690026722310252*XCR*f_del - 0.09922840349407383*XCR*f_gam - 0.070932051941581878*XNB*f_del - 1.0*XNB*f_gam - 0.1712222947342838*XNB*f_lav + 0.017223643043877631*pow(f_del, 2) + 0.19525744879034307*f_del*f_gam + 0.04125923401282721*f_del*f_lav - 0.40549970601319762*pow(f_gam, 2) - 0.046017313089315927*f_gam*f_lav);
	return fict_lav_Nb_result;

}

double s_delta()
{

	double s_delta_result;
	s_delta_result = 0.13;
	return s_delta_result;

}

double s_laves()
{

	double s_laves_result;
	s_laves_result = 0.13;
	return s_laves_result;

}

double g_gam(double XCR, double XNB)
{

	double g_gam_result;
	g_gam_result = 2267132212.7620249*pow(XCR - 0.52421634830562147, 2) + (15095482346.486225*XCR - 7913298631.5869827)*(XNB - 0.01299272922003303) + 55193083240.685936*pow(XNB - 0.01299272922003303, 2);
	return g_gam_result;

}

double g_del(double XCR, double XNB)
{

	double g_del_result;
	g_del_result = 21346492990.798885*pow(XCR - 0.022966218927631978, 2) + (16906497386.287418*XCR - 388278320.27291465)*(XNB - 0.24984563695705883) + 3085369132931.8848*pow(XNB - 0.24984563695705883, 2);
	return g_del_result;

}

double g_lav(double XCR, double XNB)
{

	double g_lav_result;
	g_lav_result = 8866730284.8069954*pow(XCR - 0.37392129441013022, 2) + (24191004361.532181*XCR - 9045531663.945219)*(XNB - 0.25826261799015571) + 98294310279.883911*pow(XNB - 0.25826261799015571, 2);
	return g_lav_result;

}

double dg_gam_dxCr(double XCR, double XNB)
{

	double dg_gam_dxCr_result;
	dg_gam_dxCr_result = 4534264425.5240498*XCR + 15095482346.486225*XNB - 2573067053.9739881;
	return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB)
{

	double dg_gam_dxNb_result;
	dg_gam_dxNb_result = 15095482346.486225*XCR + 110386166481.37187*XNB - 9347516202.3169327;
	return dg_gam_dxNb_result;

}

double dg_del_dxCr(double XCR, double XNB)
{

	double dg_del_dxCr_result;
	dg_del_dxCr_result = 42692985981.597771*XCR + 16906497386.287418*XNB - 5204511070.9175282;
	return dg_del_dxCr_result;

}

double dg_del_dxNb(double XCR, double XNB)
{

	double dg_del_dxNb_result;
	dg_del_dxNb_result = 16906497386.287418*XCR + 6170738265863.7695*XNB - 1542120310850.303;
	return dg_del_dxNb_result;

}

double dg_lav_dxCr(double XCR, double XNB)
{

	double dg_lav_dxCr_result;
	dg_lav_dxCr_result = 17733460569.613991*XCR + 24191004361.532181*XNB - 12878550648.781645;
	return dg_lav_dxCr_result;

}

double dg_lav_dxNb(double XCR, double XNB)
{

	double dg_lav_dxNb_result;
	dg_lav_dxNb_result = 24191004361.532181*XCR + 196588620559.76782*XNB - 59817023476.78421;
	return dg_lav_dxNb_result;

}

double d2g_gam_dxCrCr()
{

	double d2g_gam_dxCrCr_result;
	d2g_gam_dxCrCr_result = 4534264425.5240498;
	return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb()
{

	double d2g_gam_dxCrNb_result;
	d2g_gam_dxCrNb_result = 15095482346.486225;
	return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr()
{

	double d2g_gam_dxNbCr_result;
	d2g_gam_dxNbCr_result = 15095482346.486225;
	return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb()
{

	double d2g_gam_dxNbNb_result;
	d2g_gam_dxNbNb_result = 110386166481.37187;
	return d2g_gam_dxNbNb_result;

}

double d2g_del_dxCrCr()
{

	double d2g_del_dxCrCr_result;
	d2g_del_dxCrCr_result = 42692985981.597771;
	return d2g_del_dxCrCr_result;

}

double d2g_del_dxCrNb()
{

	double d2g_del_dxCrNb_result;
	d2g_del_dxCrNb_result = 16906497386.287418;
	return d2g_del_dxCrNb_result;

}

double d2g_del_dxNbCr()
{

	double d2g_del_dxNbCr_result;
	d2g_del_dxNbCr_result = 16906497386.287418;
	return d2g_del_dxNbCr_result;

}

double d2g_del_dxNbNb()
{

	double d2g_del_dxNbNb_result;
	d2g_del_dxNbNb_result = 6170738265863.7695;
	return d2g_del_dxNbNb_result;

}

double d2g_lav_dxCrCr()
{

	double d2g_lav_dxCrCr_result;
	d2g_lav_dxCrCr_result = 17733460569.613991;
	return d2g_lav_dxCrCr_result;

}

double d2g_lav_dxCrNb()
{

	double d2g_lav_dxCrNb_result;
	d2g_lav_dxCrNb_result = 24191004361.532181;
	return d2g_lav_dxCrNb_result;

}

double d2g_lav_dxNbCr()
{

	double d2g_lav_dxNbCr_result;
	d2g_lav_dxNbCr_result = 24191004361.532181;
	return d2g_lav_dxNbCr_result;

}

double d2g_lav_dxNbNb()
{

	double d2g_lav_dxNbNb_result;
	d2g_lav_dxNbNb_result = 196588620559.76782;
	return d2g_lav_dxNbNb_result;

}

double M_CrCr(double XCR, double XNB)
{

	double M_CrCr_result;
	M_CrCr_result = 1.0000000000000003e-15*pow(XCR, 2)*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB) + 1.0000000000000003e-15*pow(XCR, 2)*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) + 1.0000000000000003e-15*pow(1 - XCR, 2)*(8.2216387898155807e-8*XCR*(-XCR - XNB + 1) + 9.6755272489124536e-20*XCR + 3.5027570743586952e-21);
	return M_CrCr_result;

}

double M_CrNb(double XCR, double XNB)
{

	double M_CrNb_result;
	M_CrNb_result = 1.0000000000000003e-15*XCR*XNB*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) - 1.0000000000000003e-15*XCR*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*(1 - XNB) + 1.0000000000000003e-15*XNB*(1 - XCR)*(-8.2216387898155807e-8*XCR*(-XCR - XNB + 1) - 9.6755272489124536e-20*XCR - 3.5027570743586952e-21);
	return M_CrNb_result;

}

double M_NbCr(double XCR, double XNB)
{

	double M_NbCr_result;
	M_NbCr_result = 1.0000000000000003e-15*XCR*XNB*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) - 1.0000000000000003e-15*XCR*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*(1 - XNB) + 1.0000000000000003e-15*XNB*(1 - XCR)*(-8.2216387898155807e-8*XCR*(-XCR - XNB + 1) - 9.6755272489124536e-20*XCR - 3.5027570743586952e-21);
	return M_NbCr_result;

}

double M_NbNb(double XCR, double XNB)
{

	double M_NbNb_result;
	M_NbNb_result = 1.0000000000000003e-15*pow(XNB, 2)*(8.2216387898155807e-8*XCR*(-XCR - XNB + 1) + 9.6755272489124536e-20*XCR + 3.5027570743586952e-21) + 1.0000000000000003e-15*pow(XNB, 2)*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) + 1.0000000000000003e-15*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*pow(1 - XNB, 2);
	return M_NbNb_result;

}

double D_CrCr(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_CrCr_result;
	D_CrCr_result = -1.021081715345148e-11*(0.38550025800919652*pow(XCR, 4)*phi_del + 0.13334549249988586*pow(XCR, 4)*phi_lav + 0.045807617096207988*pow(XCR, 4) + 0.40379612161867279*pow(XCR, 3)*XNB*phi_del + 0.22523342638614172*pow(XCR, 3)*XNB*phi_lav + 0.19831042908880353*pow(XCR, 3)*XNB - 1.0*pow(XCR, 3)*phi_del - 0.34590247277267666*pow(XCR, 3)*phi_lav - 0.11882642396341841*pow(XCR, 3) + 0.018295863609475976*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.09188793388625531*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.15250281199259552*pow(XCR, 2)*pow(XNB, 2) - 0.64737770150468688*pow(XCR, 2)*XNB*phi_del - 0.37768106704338777*pow(XCR, 2)*XNB*phi_lav - 0.34706879901775728*pow(XCR, 2)*XNB + 0.92174961298574964*pow(XCR, 2)*phi_del + 0.31883547040902849*pow(XCR, 2)*phi_lav + 0.10952821030076157*pow(XCR, 2) - 0.014582095905157858*XCR*pow(XNB, 2)*phi_del - 0.073236152884426045*XCR*pow(XNB, 2)*phi_lav - 0.12154718015773315*XCR*pow(XNB, 2) + 0.32183196690020821*XCR*XNB*phi_del + 0.17951464302076223*XCR*XNB*phi_lav + 0.15805658359140626*XCR*XNB - 0.30724987099536938*XCR*phi_del - 0.10627849013636993*XCR*phi_lav - 0.036509403433576268*XCR + 6.2125740252721771e-16*XNB*phi_del + 3.1201620403532841e-15*XNB*phi_lav + 5.1784109719502964e-15*XNB - 1.3090111189959039e-14*phi_del - 4.5279018294761164e-15*phi_lav - 1.5554511019863601e-15);
	return D_CrCr_result;

}

double D_CrNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_CrNb_result;
	D_CrNb_result = -1.1237152145693423e-9*(0.00016624827675083329*pow(XCR, 4)*phi_del + 0.00083495433661149709*pow(XCR, 4)*phi_lav + 0.0013857410742991146*pow(XCR, 4) + 0.5564968534252428*pow(XCR, 3)*XNB*phi_del + 0.0087482013982147933*pow(XCR, 3)*XNB*phi_lav + 0.011519014156709972*pow(XCR, 3)*XNB - 0.00043125334755772673*pow(XCR, 3)*phi_del - 0.0021658982562641547*pow(XCR, 3)*phi_lav - 0.003594656671451711*pow(XCR, 3) + 0.55633060514849186*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.0079132470616032936*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.010133273082410859*pow(XCR, 2)*pow(XNB, 2) - 1.0*pow(XCR, 2)*XNB*phi_del - 0.015551175447078481*pow(XCR, 2)*XNB*phi_lav - 0.020418569389171803*pow(XCR, 2)*XNB + 0.00039750760621014357*pow(XCR, 2)*phi_del + 0.0019964158794779949*pow(XCR, 2)*phi_lav + 0.0033133733957272575*pow(XCR, 2) - 0.44340438977956459*XCR*pow(XNB, 2)*phi_del - 0.0063069844658080691*XCR*pow(XNB, 2)*phi_lav - 0.0080763807095904742*XCR*pow(XNB, 2) + 0.44353689231537335*XCR*XNB*phi_del + 0.0069724564256393728*XCR*XNB*phi_lav + 0.0091808385081728477*XCR*XNB - 0.00013250253531048005*XCR*phi_del - 0.00066547195982494148*XCR*phi_lav - 0.0011044577985745005*XCR + 1.8890855008447813e-14*XNB*phi_del + 2.6870353977177511e-16*XNB*phi_lav + 3.4408711436922738e-16*XNB - 5.645154270572686e-18*phi_del - 2.8351849000636185e-17*phi_lav - 4.7054455519030953e-17);
	return D_CrNb_result;

}

double D_NbCr(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbCr_result;
	D_NbCr_result = -7.0735449002996015e-12*(0.55647807466006949*pow(XCR, 3)*XNB*phi_del + 0.1924871420686961*pow(XCR, 3)*XNB*phi_lav + 0.066124299625904243*pow(XCR, 3)*XNB + 0.58288855492335812*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.32512938930759566*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.28626545241304863*pow(XCR, 2)*pow(XNB, 2) - 1.0*pow(XCR, 2)*XNB*phi_del - 0.34590247277267722*pow(XCR, 2)*XNB*phi_lav - 0.1188264239634185*pow(XCR, 2)*XNB + 0.02641048026328827*XCR*pow(XNB, 3)*phi_del + 0.13264224723889875*XCR*pow(XNB, 3)*phi_lav + 0.22014115278714436*XCR*pow(XNB, 3) - 0.46993240560213173*XCR*pow(XNB, 2)*phi_del - 0.28605757794273745*XCR*pow(XNB, 2)*phi_lav - 0.27284327712493966*XCR*pow(XNB, 2) + 0.4435219253391991*XCR*XNB*phi_del + 0.15341533070372734*XCR*XNB*phi_lav + 0.052702124337427159*XCR*XNB + 9.2978383732917685e-14*XCR*phi_del + 3.2161452847623037e-14*XCR*phi_lav + 1.1048288844881084e-14*XCR + 3.9942748255749408e-15*pow(XNB, 3)*phi_del + 2.0060581393155573e-14*pow(XNB, 3)*phi_lav + 3.3293762774659169e-14*pow(XNB, 3) - 1.3244233237053327e-14*pow(XNB, 2)*phi_del - 6.6516960010031681e-14*pow(XNB, 2)*phi_lav - 1.1039559839582117e-13*pow(XNB, 2) + 3.1454509740490678e-14*XNB*phi_del + 6.9609843825979855e-14*XNB*phi_lav + 1.0692631242725947e-13*XNB - 4.4127592448106699e-15*phi_del - 2.2162349829341587e-14*phi_lav - 3.6781985690547258e-14);
	return D_NbCr_result;

}

double D_NbNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbNb_result;
	D_NbNb_result = -6.2534398105391049e-10*(0.00029874073092700119*pow(XCR, 3)*XNB*phi_del + 0.0015003756651494904*pow(XCR, 3)*XNB*phi_lav + 0.0024901148421053359*pow(XCR, 3)*XNB + 1.0*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.015720127336515097*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.020699154156596476*pow(XCR, 2)*pow(XNB, 2) - 0.00053684187128035486*pow(XCR, 2)*XNB*phi_del - 0.002696199066002766*pow(XCR, 2)*XNB*phi_lav - 0.0044747761960369217*pow(XCR, 2)*XNB + 0.99970125926907305*XCR*pow(XNB, 3)*phi_del + 0.014219751671365601*XCR*pow(XNB, 3)*phi_lav + 0.018209039314491141*XCR*pow(XNB, 3) - 0.99993936041131726*XCR*pow(XNB, 2)*phi_del - 0.015415575072242706*XCR*pow(XNB, 2)*phi_lav - 0.020193700668452094*XCR*pow(XNB, 2) + 0.00023810114035296143*XCR*XNB*phi_del + 0.0011958234008512972*XCR*XNB*phi_lav + 0.0019846613539283111*XCR*XNB + 4.991468951180249e-17*XCR*phi_del + 2.5068823137913945e-16*XCR*phi_lav + 4.1605745827404651e-16*XCR + 1.5119306931137431e-13*pow(XNB, 3)*phi_del + 2.1505703630014628e-15*pow(XNB, 3)*phi_lav + 2.7539032462380823e-15*pow(XNB, 3) - 5.0132661402375525e-13*pow(XNB, 2)*phi_del - 7.1308702390517133e-15*pow(XNB, 2)*phi_lav - 9.131403946448639e-15*pow(XNB, 2) + 4.7538565139056641e-13*XNB*phi_del + 6.8126886789552162e-15*XNB*phi_lav + 8.7432727690805732e-15*XNB - 1.6703372789553427e-13*phi_del - 2.3758879059065621e-15*phi_lav - 3.0424326166394903e-15);
	return D_NbNb_result;

}
