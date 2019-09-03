/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/
#include "parabola625.cuh"
#include <math.h>

__device__ double d_h(double x)
{

	double h_result;
	h_result = x*x*x*(6.0*x*x - 15.0*x + 10.0);
	return h_result;

}

__device__ double d_hprime(double x)
{

	double hprime_result;
	hprime_result = 30.0*x*x*(1.0 - x)*(1.0 - x);
	return hprime_result;

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
	xe_gam_Cr_result = 0.55855269488699388;
	return xe_gam_Cr_result;

}

__device__ double d_xe_gam_Nb()
{

	double xe_gam_Nb_result;
	xe_gam_Nb_result = 0.010717747618334031;
	return xe_gam_Nb_result;

}

__device__ double d_xe_del_Cr()
{

	double xe_del_Cr_result;
	xe_del_Cr_result = 0.031881757028651196;
	return xe_del_Cr_result;

}

__device__ double d_xe_del_Nb()
{

	double xe_del_Nb_result;
	xe_del_Nb_result = 0.16874796371854489;
	return xe_del_Nb_result;

}

__device__ double d_xe_lav_Cr()
{

	double xe_lav_Cr_result;
	xe_lav_Cr_result = 0.44421217837672827;
	return xe_lav_Cr_result;

}

__device__ double d_xe_lav_Nb()
{

	double xe_lav_Nb_result;
	xe_lav_Nb_result = 0.17170586512707406;
	return xe_lav_Nb_result;

}

__device__ double d_matrix_min_Cr()
{

	double matrix_min_Cr_result;
	matrix_min_Cr_result = 0.27939999999999998;
	return matrix_min_Cr_result;

}

__device__ double d_matrix_max_Cr()
{

	double matrix_max_Cr_result;
	matrix_max_Cr_result = 0.32879999999999998;
	return matrix_max_Cr_result;

}

__device__ double d_matrix_min_Nb()
{

	double matrix_min_Nb_result;
	matrix_min_Nb_result = 0.021499999999999998;
	return matrix_min_Nb_result;

}

__device__ double d_matrix_max_Nb()
{

	double matrix_max_Nb_result;
	matrix_max_Nb_result = 0.0269;
	return matrix_max_Nb_result;

}

__device__ double d_enrich_min_Cr()
{

	double enrich_min_Cr_result;
	enrich_min_Cr_result = 0.24729999999999999;
	return enrich_min_Cr_result;

}

__device__ double d_enrich_max_Cr()
{

	double enrich_max_Cr_result;
	enrich_max_Cr_result = 0.29670000000000002;
	return enrich_max_Cr_result;

}

__device__ double d_enrich_min_Nb()
{

	double enrich_min_Nb_result;
	enrich_min_Nb_result = 0.16589999999999999;
	return enrich_min_Nb_result;

}

__device__ double d_enrich_max_Nb()
{

	double enrich_max_Nb_result;
	enrich_max_Nb_result = 0.1726;
	return enrich_max_Nb_result;

}

__device__ double d_xr_gam_Cr(double P_del, double P_lav)
{

	double xr_gam_Cr_result;
	xr_gam_Cr_result = -9.6984079989082258e-10*P_del + 6.3118903213767551e-10*P_lav + 0.55855269488699388;
	return xr_gam_Cr_result;

}

__device__ double d_xr_gam_Nb(double P_del, double P_lav)
{

	double xr_gam_Nb_result;
	xr_gam_Nb_result = 1.0048047396144418e-10*P_del - 1.1106547204101917e-11*P_lav + 0.010717747618334031;
	return xr_gam_Nb_result;

}

__device__ double d_xr_del_Cr(double P_del, double P_lav)
{

	double xr_del_Cr_result;
	xr_del_Cr_result = -6.6194274436412276e-11*P_del - 1.5374687132506466e-10*P_lav + 0.031881757028651196;
	return xr_del_Cr_result;

}

__device__ double d_xr_del_Nb(double P_del, double P_lav)
{

	double xr_del_Nb_result;
	xr_del_Nb_result = -1.9581600075232547e-11*P_del + 3.9815435461767849e-10*P_lav + 0.16874796371854489;
	return xr_del_Nb_result;

}

__device__ double d_xr_lav_Cr(double P_del, double P_lav)
{

	double xr_lav_Cr_result;
	xr_lav_Cr_result = -1.2953044143338369e-9*P_del + 4.3416027339639082e-10*P_lav + 0.44421217837672827;
	return xr_lav_Cr_result;

}

__device__ double d_xr_lav_Nb(double P_del, double P_lav)
{

	double xr_lav_Nb_result;
	xr_lav_Nb_result = 2.6966712244807525e-10*P_del - 6.3173100035954083e-11*P_lav + 0.17170586512707406;
	return xr_lav_Nb_result;

}

__device__ double d_inv_fict_det(double f_del, double f_gam, double f_lav)
{

	double inv_fict_det_result;
	inv_fict_det_result = 10.511929918754895/(0.063619913592430452*f_del*f_del + 0.77472418933378007*f_del*f_gam + 1.0*f_del*f_lav + 0.11406419308318602*f_gam*f_gam + 0.29936120577567832*f_gam*f_lav + 0.076367679948657952*f_lav*f_lav);
	return inv_fict_det_result;

}

__device__ double d_fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_gam_Cr_result;
	fict_gam_Cr_result = 0.081719161132759205*INV_DET*(0.94267830414080844*XCR*f_del + 0.13278315265091145*XCR*f_gam + 0.27475106182917869*XCR*f_lav + 0.48142139214471374*XNB*f_del + 1.0*XNB*f_lav - 0.069926420929843164*f_del*f_del - 0.03219039130480722*f_del*f_gam - 0.028703356204519043*f_del*f_lav - 0.028515077359655715*f_gam*f_lav - 0.24409812910594283*f_lav*f_lav);
	return fict_gam_Cr_result;

}

__device__ double d_fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_gam_Nb_result;
	fict_gam_Nb_result = -0.010850927847195595*INV_DET*(0.75572285002798056*XCR*f_del - 0.063676254673428662*XCR*f_lav + 0.30738018673156009*XNB*f_del - 1.0*XNB*f_gam - 0.55532654346045196*XNB*f_lav - 0.081941433925801291*f_del*f_del - 0.32945245368491854*f_del*f_gam - 0.38670236006564801*f_del*f_lav + 0.18509555516266776*f_gam*f_lav + 0.11646290035390382*f_lav*f_lav);
	return fict_gam_Nb_result;

}

__device__ double d_fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_del_Cr_result;
	fict_del_Cr_result = 0.046966148206032843*INV_DET*(0.12886223924609808*XCR*f_del - 0.07101626075977073*XCR*f_gam - 0.20028342359736442*XCR*f_lav - 0.83765336993033834*XNB*f_gam - 1.0*XNB*f_lav + 0.12166908885811056*f_del*f_gam + 0.18246772627205363*f_del*f_lav + 0.056009953433358468*f_gam*f_gam + 0.31729459226818346*f_gam*f_lav + 0.26560576240348166*f_lav*f_lav);
	return fict_del_Cr_result;

}

__device__ double d_fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_del_Nb_result;
	fict_del_Nb_result = 0.10453655111851477*INV_DET*(0.078444276479274974*XCR*f_gam + 0.20923881653433052*XCR*f_lav + 0.057895185577072183*XNB*f_del + 0.73691813445330934*XNB*f_gam + 1.0*XNB*f_lav - 0.0085055473679878923*f_del*f_gam - 0.031796338023487768*f_del*f_lav - 0.034197271344486177*f_gam*f_gam - 0.24299677574697207*f_gam*f_lav - 0.25292500971253778*f_lav*f_lav);
	return fict_del_Nb_result;

}

__device__ double d_fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_lav_Cr_result;
	fict_lav_Cr_result = 0.10453655111851476*INV_DET*(1.0*XCR*f_del + 0.057643074984273575*XCR*f_gam + 0.0694958662006743*XCR*f_lav + 0.44927967972452598*XNB*f_del - 0.78172811574884349*XNB*f_gam - 0.081979041619570731*f_del*f_del - 0.12011579223114696*f_del*f_gam - 0.11933127186562477*f_del*f_lav + 0.022291037694796172*f_gam*f_gam + 0.19081837052380662*f_gam*f_lav);
	return fict_lav_Cr_result;

}

__device__ double d_fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_lav_Nb_result;
	fict_lav_Nb_result = -0.02643990820696555*INV_DET*(0.82727610358556924*XCR*f_del + 0.026132709676314116*XCR*f_gam + 0.35577056025509707*XNB*f_del - 0.84918699858043711*XNB*f_gam - 0.27476866079657325*XNB*f_lav - 0.12571448770378987*f_del*f_del - 0.80204383828035919*f_del*f_gam - 1.0*f_del*f_lav - 0.075963142465736502*f_gam*f_gam - 0.047796328138628434*f_gam*f_lav);
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
	g_gam_result = 2005672715.5837781*pow(XCR - 0.55855269488699388, 2) + (14703563128.545319*XCR - 8212714809.8900261)*(XNB - 0.010717747618334031) + 62431733279.319511*pow(XNB - 0.010717747618334031, 2);
	return g_gam_result;

}

__device__ double d_g_del(double XCR, double XNB)
{

	double g_del_result;
	g_del_result = 15567985511.489613*pow(XCR - 0.031881757028651196, 2) + (17972099186.595295*XCR - 572982099.56185102)*(XNB - 0.16874796371854489) + 13383100915.719385*pow(XNB - 0.16874796371854489, 2);
	return g_del_result;

}

__device__ double d_g_lav(double XCR, double XNB)
{

	double g_lav_result;
	g_lav_result = 6897850581.6836567*pow(XCR - 0.44421217837672827, 2) + (57317655210.986305*XCR - 25461200480.718456)*(XNB - 0.17170586512707406) + 134480681274.87074*pow(XNB - 0.17170586512707406, 2);
	return g_lav_result;

}

__device__ double d_dg_gam_dxCr(double XCR, double XNB)
{

	double dg_gam_dxCr_result;
	dg_gam_dxCr_result = 4011345431.1675563*XCR + 14703563128.545319*XNB - 2398136879.4032598;
	return dg_gam_dxCr_result;

}

__device__ double d_dg_gam_dxNb(double XCR, double XNB)
{

	double dg_gam_dxNb_result;
	dg_gam_dxNb_result = 14703563128.545319*XCR + 124863466558.63902*XNB - 9550969931.2158108;
	return dg_gam_dxNb_result;

}

__device__ double d_dg_del_dxCr(double XCR, double XNB)
{

	double dg_del_dxCr_result;
	dg_del_dxCr_result = 31135971022.979225*XCR + 17972099186.595295*XNB - 4025424604.4914207;
	return dg_del_dxCr_result;

}

__device__ double d_dg_del_dxNb(double XCR, double XNB)
{

	double dg_del_dxNb_result;
	dg_del_dxNb_result = 17972099186.595295*XCR + 26766201831.43877*XNB - 5089724155.0967312;
	return dg_del_dxNb_result;

}

__device__ double d_dg_lav_dxCr(double XCR, double XNB)
{

	double dg_lav_dxCr_result;
	dg_lav_dxCr_result = 13795701163.367313*XCR + 57317655210.986305*XNB - 15969996041.071507;
	return dg_lav_dxCr_result;

}

__device__ double d_dg_lav_dxNb(double XCR, double XNB)
{

	double dg_lav_dxNb_result;
	dg_lav_dxNb_result = 57317655210.986305*XCR + 268961362549.74149*XNB - 71643443923.07843;
	return dg_lav_dxNb_result;

}

__device__ double d_d2g_gam_dxCrCr()
{

	double d2g_gam_dxCrCr_result;
	d2g_gam_dxCrCr_result = 4011345431.1675563;
	return d2g_gam_dxCrCr_result;

}

__device__ double d_d2g_gam_dxCrNb()
{

	double d2g_gam_dxCrNb_result;
	d2g_gam_dxCrNb_result = 14703563128.545319;
	return d2g_gam_dxCrNb_result;

}

__device__ double d_d2g_gam_dxNbCr()
{

	double d2g_gam_dxNbCr_result;
	d2g_gam_dxNbCr_result = 14703563128.545319;
	return d2g_gam_dxNbCr_result;

}

__device__ double d_d2g_gam_dxNbNb()
{

	double d2g_gam_dxNbNb_result;
	d2g_gam_dxNbNb_result = 124863466558.63902;
	return d2g_gam_dxNbNb_result;

}

__device__ double d_d2g_del_dxCrCr()
{

	double d2g_del_dxCrCr_result;
	d2g_del_dxCrCr_result = 31135971022.979225;
	return d2g_del_dxCrCr_result;

}

__device__ double d_d2g_del_dxCrNb()
{

	double d2g_del_dxCrNb_result;
	d2g_del_dxCrNb_result = 17972099186.595295;
	return d2g_del_dxCrNb_result;

}

__device__ double d_d2g_del_dxNbCr()
{

	double d2g_del_dxNbCr_result;
	d2g_del_dxNbCr_result = 17972099186.595295;
	return d2g_del_dxNbCr_result;

}

__device__ double d_d2g_del_dxNbNb()
{

	double d2g_del_dxNbNb_result;
	d2g_del_dxNbNb_result = 26766201831.43877;
	return d2g_del_dxNbNb_result;

}

__device__ double d_d2g_lav_dxCrCr()
{

	double d2g_lav_dxCrCr_result;
	d2g_lav_dxCrCr_result = 13795701163.367313;
	return d2g_lav_dxCrCr_result;

}

__device__ double d_d2g_lav_dxCrNb()
{

	double d2g_lav_dxCrNb_result;
	d2g_lav_dxCrNb_result = 57317655210.986305;
	return d2g_lav_dxCrNb_result;

}

__device__ double d_d2g_lav_dxNbCr()
{

	double d2g_lav_dxNbCr_result;
	d2g_lav_dxNbCr_result = 57317655210.986305;
	return d2g_lav_dxNbCr_result;

}

__device__ double d_d2g_lav_dxNbNb()
{

	double d2g_lav_dxNbNb_result;
	d2g_lav_dxNbNb_result = 268961362549.74149;
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

__device__ double d_D_CrCr(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_CrCr_result;
	D_CrCr_result = -7.2582251443309983e-12*(0.38550025800919668*pow(XCR, 4)*phi_del + 0.38550025800919668*pow(XCR, 4)*phi_lav + 0.057009992393993084*pow(XCR, 4) + 0.43195330450105324*pow(XCR, 3)*XNB*phi_del + 0.43195330450105324*pow(XCR, 3)*XNB*phi_lav + 0.26597978482570345*pow(XCR, 3)*XNB - 1.0*pow(XCR, 3)*phi_del - 1.0*pow(XCR, 3)*phi_lav - 0.14788574380833097*pow(XCR, 3) + 0.046453046491856338*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.046453046491856338*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.20896979243170991*pow(XCR, 2)*pow(XNB, 2) - 0.69797660946629991*pow(XCR, 2)*XNB*phi_del - 0.69797660946629991*pow(XCR, 2)*XNB*phi_lav - 0.46639781049868945*pow(XCR, 2)*XNB + 0.92174961298574987*pow(XCR, 2)*phi_del + 0.92174961298574987*pow(XCR, 2)*phi_lav + 0.13631362712143877*pow(XCR, 2) - 0.037023820984332906*XCR*pow(XNB, 2)*phi_del - 0.037023820984332906*XCR*pow(XNB, 2)*phi_lav - 0.16655226665233169*XCR*pow(XNB, 2) + 0.34427369197940372*XCR*XNB*phi_del + 0.34427369197940372*XCR*XNB*phi_lav + 0.21199014235959868*XCR*XNB - 0.30724987099536466*XCR*phi_del - 0.30724987099536466*XCR*phi_lav - 0.045437875707129448*XCR + 1.5773674104161867e-15*XNB*phi_del + 1.5773674104161867e-15*XNB*phi_lav + 7.0958131971172007e-15*XNB - 1.3090111189959037e-14*phi_del - 1.3090111189959037e-14*phi_lav - 1.9358408298608479e-15);
	return D_CrCr_result;

}

__device__ double d_D_CrNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_CrNb_result;
	D_CrNb_result = -2.5563893615445794e-11*(0.013189175137008522*pow(XCR, 4)*phi_del + 0.013189175137008522*pow(XCR, 4)*phi_lav + 0.059331720928342505*pow(XCR, 4) - 0.38265226028544108*pow(XCR, 3)*XNB*phi_del - 0.38265226028544108*pow(XCR, 3)*XNB*phi_lav + 0.56317995738880211*pow(XCR, 3)*XNB - 0.034213142178218539*pow(XCR, 3)*phi_del - 0.034213142178218539*pow(XCR, 3)*phi_lav - 0.15390838188992101*pow(XCR, 3) - 0.39584143542244965*pow(XCR, 2)*pow(XNB, 2)*phi_del - 0.39584143542244965*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.50384823646045962*pow(XCR, 2)*pow(XNB, 2) + 0.69030942316337007*pow(XCR, 2)*XNB*phi_del + 0.69030942316337007*pow(XCR, 2)*XNB*phi_lav - 1.0*pow(XCR, 2)*XNB + 0.031535950561799343*pow(XCR, 2)*phi_del + 0.031535950561799343*pow(XCR, 2)*phi_lav + 0.14186499144229758*pow(XCR, 2) + 0.31549195478129444*XCR*pow(XNB, 2)*phi_del + 0.31549195478129444*XCR*pow(XNB, 2)*phi_lav - 0.40157510257705348*XCR*pow(XNB, 2) - 0.30497997126099063*XCR*XNB*phi_del - 0.30497997126099063*XCR*XNB*phi_lav + 0.44886343305815379*XCR*XNB - 0.010511983520670105*XCR*phi_del - 0.010511983520670105*XCR*phi_lav - 0.047288330480700545*XCR - 1.3441257938533574e-14*XNB*phi_del - 1.3441257938533574e-14*XNB*phi_lav + 1.7108754925852337e-14*XNB - 4.4785383527075794e-16*phi_del - 4.4785383527075794e-16*phi_lav - 2.0146778320057434e-15);
	return D_CrNb_result;

}

__device__ double d_D_NbCr(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbCr_result;
	D_NbCr_result = -5.0281364050823669e-12*(0.55647807466006949*pow(XCR, 3)*XNB*phi_del + 0.55647807466006949*pow(XCR, 3)*XNB*phi_lav + 0.082295173984131637*pow(XCR, 3)*XNB + 0.6235340657698506*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.6235340657698506*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.38394765109281975*pow(XCR, 2)*pow(XNB, 2) - 1.0*pow(XCR, 2)*XNB*phi_del - 1.0*pow(XCR, 2)*XNB*phi_lav - 0.14788574380833081*pow(XCR, 2)*XNB + 0.067055991109780833*XCR*pow(XNB, 3)*phi_del + 0.067055991109780833*XCR*pow(XNB, 3)*phi_lav + 0.30165247710868742*XCR*pow(XNB, 3) - 0.51057791644870132*XCR*pow(XNB, 2)*phi_del - 0.51057791644870132*XCR*pow(XNB, 2)*phi_lav - 0.36724304693328863*XCR*pow(XNB, 2) + 0.44352192533920265*XCR*XNB*phi_del + 0.44352192533920265*XCR*XNB*phi_lav + 0.065590569824090533*XCR*XNB + 9.297838373291766e-14*XCR*phi_del + 9.297838373291766e-14*XCR*phi_lav + 1.3750177436438941e-14*XCR + 1.0141430770044922e-14*pow(XNB, 3)*phi_del + 1.0141430770044922e-14*pow(XNB, 3)*phi_lav + 4.5621392847686756e-14*pow(XNB, 3) - 3.3626998727252252e-14*pow(XNB, 2)*phi_del - 3.3626998727252252e-14*pow(XNB, 2)*phi_lav - 1.5127160595090692e-13*pow(XNB, 2) + 5.0782164036196637e-14*XNB*phi_del + 5.0782164036196637e-14*XNB*phi_lav + 1.4623547970025454e-13*XNB - 1.1203959251772693e-14*phi_del - 1.1203959251772693e-14*phi_lav - 5.0401194670121737e-14);
	return D_NbCr_result;

}

__device__ double d_D_NbNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbNb_result;
	D_NbNb_result = -1.4397072517038633e-11*(0.023419113134211061*pow(XCR, 3)*XNB*phi_del + 0.023419113134211061*pow(XCR, 3)*XNB*phi_lav + 0.10535126499074204*pow(XCR, 3)*XNB - 0.67944935764336822*pow(XCR, 2)*pow(XNB, 2)*phi_del - 0.67944935764336822*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 1.0*pow(XCR, 2)*pow(XNB, 2) - 0.042084520847505039*pow(XCR, 2)*XNB*phi_del - 0.042084520847505039*pow(XCR, 2)*XNB*phi_lav - 0.18931790808666982*pow(XCR, 2)*XNB - 0.7028684707775793*XCR*pow(XNB, 3)*phi_del - 0.7028684707775793*XCR*pow(XNB, 3)*phi_lav + 0.8946487350092579*XCR*pow(XNB, 3) + 0.68420306306566314*XCR*pow(XNB, 2)*phi_del + 0.68420306306566314*XCR*pow(XNB, 2)*phi_lav - 0.97861537810666321*XCR*pow(XNB, 2) + 0.018665407713263051*XCR*XNB*phi_del + 0.018665407713263051*XCR*XNB*phi_lav + 0.083966643095789342*XCR*XNB + 3.9129507285752948e-15*XCR*phi_del + 3.9129507285752948e-15*XCR*phi_lav + 1.7602473105595626e-14*XCR - 1.0630059773732023e-13*pow(XNB, 3)*phi_del - 1.0630059773732023e-13*pow(XNB, 3)*phi_lav + 1.3530510934885312e-13*pow(XNB, 3) + 3.524719682924176e-13*pow(XNB, 2)*phi_del + 3.524719682924176e-13*pow(XNB, 2)*phi_lav - 4.486452496726414e-13*pow(XNB, 2) - 3.334310795517053e-13*XNB*phi_del - 3.334310795517053e-13*XNB*phi_lav + 4.2899851205846871e-13*XNB + 1.1743782435569896e-13*phi_del + 1.1743782435569896e-13*phi_lav - 1.4948116947945047e-13);
	return D_NbNb_result;

}
