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
	xe_gam_Cr_result = 0.55855269488699388;
	return xe_gam_Cr_result;

}

double xe_gam_Nb()
{

	double xe_gam_Nb_result;
	xe_gam_Nb_result = 0.010717747618334031;
	return xe_gam_Nb_result;

}

double xe_del_Cr()
{

	double xe_del_Cr_result;
	xe_del_Cr_result = 0.031881757028651196;
	return xe_del_Cr_result;

}

double xe_del_Nb()
{

	double xe_del_Nb_result;
	xe_del_Nb_result = 0.16874796371854489;
	return xe_del_Nb_result;

}

double xe_lav_Cr()
{

	double xe_lav_Cr_result;
	xe_lav_Cr_result = 0.44421217837672827;
	return xe_lav_Cr_result;

}

double xe_lav_Nb()
{

	double xe_lav_Nb_result;
	xe_lav_Nb_result = 0.17170586512707406;
	return xe_lav_Nb_result;

}

double matrix_min_Cr()
{

	double matrix_min_Cr_result;
	matrix_min_Cr_result = 0.27939999999999998;
	return matrix_min_Cr_result;

}

double matrix_max_Cr()
{

	double matrix_max_Cr_result;
	matrix_max_Cr_result = 0.32879999999999998;
	return matrix_max_Cr_result;

}

double matrix_min_Nb()
{

	double matrix_min_Nb_result;
	matrix_min_Nb_result = 0.021499999999999998;
	return matrix_min_Nb_result;

}

double matrix_max_Nb()
{

	double matrix_max_Nb_result;
	matrix_max_Nb_result = 0.0269;
	return matrix_max_Nb_result;

}

double enrich_min_Cr()
{

	double enrich_min_Cr_result;
	enrich_min_Cr_result = 0.24729999999999999;
	return enrich_min_Cr_result;

}

double enrich_max_Cr()
{

	double enrich_max_Cr_result;
	enrich_max_Cr_result = 0.29670000000000002;
	return enrich_max_Cr_result;

}

double enrich_min_Nb()
{

	double enrich_min_Nb_result;
	enrich_min_Nb_result = 0.16589999999999999;
	return enrich_min_Nb_result;

}

double enrich_max_Nb()
{

	double enrich_max_Nb_result;
	enrich_max_Nb_result = 0.1726;
	return enrich_max_Nb_result;

}

double xr_gam_Cr(double P_del, double P_lav)
{

	double xr_gam_Cr_result;
	xr_gam_Cr_result = -9.6984079989082258e-10*P_del + 6.3118903213767551e-10*P_lav + 0.55855269488699388;
	return xr_gam_Cr_result;

}

double xr_gam_Nb(double P_del, double P_lav)
{

	double xr_gam_Nb_result;
	xr_gam_Nb_result = 1.0048047396144418e-10*P_del - 1.1106547204101917e-11*P_lav + 0.010717747618334031;
	return xr_gam_Nb_result;

}

double xr_del_Cr(double P_del, double P_lav)
{

	double xr_del_Cr_result;
	xr_del_Cr_result = -6.6194274436412276e-11*P_del - 1.5374687132506466e-10*P_lav + 0.031881757028651196;
	return xr_del_Cr_result;

}

double xr_del_Nb(double P_del, double P_lav)
{

	double xr_del_Nb_result;
	xr_del_Nb_result = -1.9581600075232547e-11*P_del + 3.9815435461767849e-10*P_lav + 0.16874796371854489;
	return xr_del_Nb_result;

}

double xr_lav_Cr(double P_del, double P_lav)
{

	double xr_lav_Cr_result;
	xr_lav_Cr_result = -1.2953044143338369e-9*P_del + 4.3416027339639082e-10*P_lav + 0.44421217837672827;
	return xr_lav_Cr_result;

}

double xr_lav_Nb(double P_del, double P_lav)
{

	double xr_lav_Nb_result;
	xr_lav_Nb_result = 2.6966712244807525e-10*P_del - 6.3173100035954083e-11*P_lav + 0.17170586512707406;
	return xr_lav_Nb_result;

}

double inv_fict_det(double f_del, double f_gam, double f_lav)
{

	double inv_fict_det_result;
	inv_fict_det_result = 10.511929918754895/(0.063619913592430452*pow(f_del, 2) + 0.77472418933378007*f_del*f_gam + 1.0*f_del*f_lav + 0.11406419308318602*pow(f_gam, 2) + 0.29936120577567832*f_gam*f_lav + 0.076367679948657952*pow(f_lav, 2));
	return inv_fict_det_result;

}

double fict_gam_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_gam_Cr_result;
	fict_gam_Cr_result = 0.081719161132759205*INV_DET*(0.94267830414080844*XCR*f_del + 0.13278315265091145*XCR*f_gam + 0.27475106182917869*XCR*f_lav + 0.48142139214471374*XNB*f_del + 1.0*XNB*f_lav - 0.069926420929843164*pow(f_del, 2) - 0.03219039130480722*f_del*f_gam - 0.028703356204519043*f_del*f_lav - 0.028515077359655715*f_gam*f_lav - 0.24409812910594283*pow(f_lav, 2));
	return fict_gam_Cr_result;

}

double fict_gam_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_gam_Nb_result;
	fict_gam_Nb_result = -0.010850927847195595*INV_DET*(0.75572285002798056*XCR*f_del - 0.063676254673428662*XCR*f_lav + 0.30738018673156009*XNB*f_del - 1.0*XNB*f_gam - 0.55532654346045196*XNB*f_lav - 0.081941433925801291*pow(f_del, 2) - 0.32945245368491854*f_del*f_gam - 0.38670236006564801*f_del*f_lav + 0.18509555516266776*f_gam*f_lav + 0.11646290035390382*pow(f_lav, 2));
	return fict_gam_Nb_result;

}

double fict_del_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_del_Cr_result;
	fict_del_Cr_result = 0.046966148206032843*INV_DET*(0.12886223924609808*XCR*f_del - 0.07101626075977073*XCR*f_gam - 0.20028342359736442*XCR*f_lav - 0.83765336993033834*XNB*f_gam - 1.0*XNB*f_lav + 0.12166908885811056*f_del*f_gam + 0.18246772627205363*f_del*f_lav + 0.056009953433358468*pow(f_gam, 2) + 0.31729459226818346*f_gam*f_lav + 0.26560576240348166*pow(f_lav, 2));
	return fict_del_Cr_result;

}

double fict_del_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_del_Nb_result;
	fict_del_Nb_result = 0.10453655111851477*INV_DET*(0.078444276479274974*XCR*f_gam + 0.20923881653433052*XCR*f_lav + 0.057895185577072183*XNB*f_del + 0.73691813445330934*XNB*f_gam + 1.0*XNB*f_lav - 0.0085055473679878923*f_del*f_gam - 0.031796338023487768*f_del*f_lav - 0.034197271344486177*pow(f_gam, 2) - 0.24299677574697207*f_gam*f_lav - 0.25292500971253778*pow(f_lav, 2));
	return fict_del_Nb_result;

}

double fict_lav_Cr(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_lav_Cr_result;
	fict_lav_Cr_result = 0.10453655111851476*INV_DET*(1.0*XCR*f_del + 0.057643074984273575*XCR*f_gam + 0.0694958662006743*XCR*f_lav + 0.44927967972452598*XNB*f_del - 0.78172811574884349*XNB*f_gam - 0.081979041619570731*pow(f_del, 2) - 0.12011579223114696*f_del*f_gam - 0.11933127186562477*f_del*f_lav + 0.022291037694796172*pow(f_gam, 2) + 0.19081837052380662*f_gam*f_lav);
	return fict_lav_Cr_result;

}

double fict_lav_Nb(double INV_DET, double XCR, double XNB, double f_del, double f_gam, double f_lav)
{

	double fict_lav_Nb_result;
	fict_lav_Nb_result = -0.02643990820696555*INV_DET*(0.82727610358556924*XCR*f_del + 0.026132709676314116*XCR*f_gam + 0.35577056025509707*XNB*f_del - 0.84918699858043711*XNB*f_gam - 0.27476866079657325*XNB*f_lav - 0.12571448770378987*pow(f_del, 2) - 0.80204383828035919*f_del*f_gam - 1.0*f_del*f_lav - 0.075963142465736502*pow(f_gam, 2) - 0.047796328138628434*f_gam*f_lav);
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
	g_gam_result = 2005672715.5837781*pow(XCR - 0.55855269488699388, 2) + (14703563128.545319*XCR - 8212714809.8900261)*(XNB - 0.010717747618334031) + 62431733279.319511*pow(XNB - 0.010717747618334031, 2);
	return g_gam_result;

}

double g_del(double XCR, double XNB)
{

	double g_del_result;
	g_del_result = 15567985511.489613*pow(XCR - 0.031881757028651196, 2) + (17972099186.595295*XCR - 572982099.56185102)*(XNB - 0.16874796371854489) + 13383100915.719385*pow(XNB - 0.16874796371854489, 2);
	return g_del_result;

}

double g_lav(double XCR, double XNB)
{

	double g_lav_result;
	g_lav_result = 6897850581.6836567*pow(XCR - 0.44421217837672827, 2) + (57317655210.986305*XCR - 25461200480.718456)*(XNB - 0.17170586512707406) + 134480681274.87074*pow(XNB - 0.17170586512707406, 2);
	return g_lav_result;

}

double dg_gam_dxCr(double XCR, double XNB)
{

	double dg_gam_dxCr_result;
	dg_gam_dxCr_result = 4011345431.1675563*XCR + 14703563128.545319*XNB - 2398136879.4032598;
	return dg_gam_dxCr_result;

}

double dg_gam_dxNb(double XCR, double XNB)
{

	double dg_gam_dxNb_result;
	dg_gam_dxNb_result = 14703563128.545319*XCR + 124863466558.63902*XNB - 9550969931.2158108;
	return dg_gam_dxNb_result;

}

double dg_del_dxCr(double XCR, double XNB)
{

	double dg_del_dxCr_result;
	dg_del_dxCr_result = 31135971022.979225*XCR + 17972099186.595295*XNB - 4025424604.4914207;
	return dg_del_dxCr_result;

}

double dg_del_dxNb(double XCR, double XNB)
{

	double dg_del_dxNb_result;
	dg_del_dxNb_result = 17972099186.595295*XCR + 26766201831.43877*XNB - 5089724155.0967312;
	return dg_del_dxNb_result;

}

double dg_lav_dxCr(double XCR, double XNB)
{

	double dg_lav_dxCr_result;
	dg_lav_dxCr_result = 13795701163.367313*XCR + 57317655210.986305*XNB - 15969996041.071507;
	return dg_lav_dxCr_result;

}

double dg_lav_dxNb(double XCR, double XNB)
{

	double dg_lav_dxNb_result;
	dg_lav_dxNb_result = 57317655210.986305*XCR + 268961362549.74149*XNB - 71643443923.07843;
	return dg_lav_dxNb_result;

}

double d2g_gam_dxCrCr()
{

	double d2g_gam_dxCrCr_result;
	d2g_gam_dxCrCr_result = 4011345431.1675563;
	return d2g_gam_dxCrCr_result;

}

double d2g_gam_dxCrNb()
{

	double d2g_gam_dxCrNb_result;
	d2g_gam_dxCrNb_result = 14703563128.545319;
	return d2g_gam_dxCrNb_result;

}

double d2g_gam_dxNbCr()
{

	double d2g_gam_dxNbCr_result;
	d2g_gam_dxNbCr_result = 14703563128.545319;
	return d2g_gam_dxNbCr_result;

}

double d2g_gam_dxNbNb()
{

	double d2g_gam_dxNbNb_result;
	d2g_gam_dxNbNb_result = 124863466558.63902;
	return d2g_gam_dxNbNb_result;

}

double d2g_del_dxCrCr()
{

	double d2g_del_dxCrCr_result;
	d2g_del_dxCrCr_result = 31135971022.979225;
	return d2g_del_dxCrCr_result;

}

double d2g_del_dxCrNb()
{

	double d2g_del_dxCrNb_result;
	d2g_del_dxCrNb_result = 17972099186.595295;
	return d2g_del_dxCrNb_result;

}

double d2g_del_dxNbCr()
{

	double d2g_del_dxNbCr_result;
	d2g_del_dxNbCr_result = 17972099186.595295;
	return d2g_del_dxNbCr_result;

}

double d2g_del_dxNbNb()
{

	double d2g_del_dxNbNb_result;
	d2g_del_dxNbNb_result = 26766201831.43877;
	return d2g_del_dxNbNb_result;

}

double d2g_lav_dxCrCr()
{

	double d2g_lav_dxCrCr_result;
	d2g_lav_dxCrCr_result = 13795701163.367313;
	return d2g_lav_dxCrCr_result;

}

double d2g_lav_dxCrNb()
{

	double d2g_lav_dxCrNb_result;
	d2g_lav_dxCrNb_result = 57317655210.986305;
	return d2g_lav_dxCrNb_result;

}

double d2g_lav_dxNbCr()
{

	double d2g_lav_dxNbCr_result;
	d2g_lav_dxNbCr_result = 57317655210.986305;
	return d2g_lav_dxNbCr_result;

}

double d2g_lav_dxNbNb()
{

	double d2g_lav_dxNbNb_result;
	d2g_lav_dxNbNb_result = 268961362549.74149;
	return d2g_lav_dxNbNb_result;

}

double M_CrCr(double XCR, double XNB)
{

	double M_CrCr_result;
	M_CrCr_result = pow(XCR, 2)*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB) + pow(XCR, 2)*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) + pow(1 - XCR, 2)*(8.2216387898155807e-8*XCR*(-XCR - XNB + 1) + 9.6755272489124536e-20*XCR + 3.5027570743586952e-21);
	return M_CrCr_result;

}

double M_CrNb(double XCR, double XNB)
{

	double M_CrNb_result;
	M_CrNb_result = XCR*XNB*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) - XCR*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*(1 - XNB) + XNB*(1 - XCR)*(-8.2216387898155807e-8*XCR*(-XCR - XNB + 1) - 9.6755272489124536e-20*XCR - 3.5027570743586952e-21);
	return M_CrNb_result;

}

double M_NbCr(double XCR, double XNB)
{

	double M_NbCr_result;
	M_NbCr_result = XCR*XNB*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) - XCR*(1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*(1 - XNB) + XNB*(1 - XCR)*(-8.2216387898155807e-8*XCR*(-XCR - XNB + 1) - 9.6755272489124536e-20*XCR - 3.5027570743586952e-21);
	return M_NbCr_result;

}

double M_NbNb(double XCR, double XNB)
{

	double M_NbNb_result;
	M_NbNb_result = pow(XNB, 2)*(8.2216387898155807e-8*XCR*(-XCR - XNB + 1) + 9.6755272489124536e-20*XCR + 3.5027570743586952e-21) + pow(XNB, 2)*(2.0938866959006431e-8*XCR*(-XCR - XNB + 1) + 9.8428461923389931e-20*XCR - 1.0199962450633582e-21*XNB + 1.8295676400933012e-21) + (1.7235555733323437e-20 - 1.4581024012583029e-20*XNB)*pow(1 - XNB, 2);
	return M_NbNb_result;

}

double D_CrCr(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_CrCr_result;
	D_CrCr_result = -2.7980476658284387e-12*pow(XCR, 4)*phi_del - 2.7980476658284387e-12*pow(XCR, 4)*phi_lav - 4.1379136027220276e-13*pow(XCR, 4) - 3.1352143359064083e-12*pow(XCR, 3)*XNB*phi_del - 3.1352143359064083e-12*pow(XCR, 3)*XNB*phi_lav - 1.9305411621056693e-12*pow(XCR, 3)*XNB + 7.2582251443309991e-12*pow(XCR, 3)*phi_del + 7.2582251443309991e-12*pow(XCR, 3)*phi_lav + 1.0733880241977198e-12*pow(XCR, 3) - 3.3716667007796982e-13*pow(XCR, 2)*pow(XNB, 2)*phi_del - 3.3716667007796982e-13*pow(XCR, 2)*pow(XNB, 2)*phi_lav - 1.5167498018334665e-12*pow(XCR, 2)*pow(XNB, 2) + 5.0660713769831958e-12*pow(XCR, 2)*XNB*phi_del + 5.0660713769831958e-12*pow(XCR, 2)*XNB*phi_lav + 3.3852203154225117e-12*pow(XCR, 2)*XNB - 6.6902662177505352e-12*pow(XCR, 2)*phi_del - 6.6902662177505352e-12*pow(XCR, 2)*phi_lav - 9.8939499588778664e-13*pow(XCR, 2) + 2.6872722840769506e-13*XCR*pow(XNB, 2)*phi_del + 2.6872722840769506e-13*XCR*pow(XNB, 2)*phi_lav + 1.2088738496612743e-12*XCR*pow(XNB, 2) - 2.4988159676565734e-12*XCR*XNB*phi_del - 2.4988159676565734e-12*XCR*XNB*phi_lav - 1.5386721816247473e-12*XCR*XNB + 2.2300887392510112e-12*XCR*phi_del + 2.2300887392510112e-12*XCR*phi_lav + 3.2979833196247364e-13*XCR - 1.1448887800131043e-26*XNB*phi_del - 1.1448887800131043e-26*XNB*phi_lav - 5.1503009766791795e-26*XNB + 9.5010974181049266e-26*phi_del + 9.5010974181049266e-26*phi_lav + 1.4050768586718594e-26;
	return D_CrCr_result;

}

double D_CrNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_CrNb_result;
	D_CrNb_result = -3.3716667007796982e-13*pow(XCR, 4)*phi_del - 3.3716667007796982e-13*pow(XCR, 4)*phi_lav - 1.5167498018334665e-12*pow(XCR, 4) + 9.7820816736468872e-12*pow(XCR, 3)*XNB*phi_del + 9.7820816736468872e-12*pow(XCR, 3)*XNB*phi_lav - 1.4397072517038633e-11*pow(XCR, 3)*XNB + 8.7462112689409982e-13*pow(XCR, 3)*phi_del + 8.7462112689409982e-13*pow(XCR, 3)*phi_lav + 3.9344975011593433e-12*pow(XCR, 3) + 1.0119248343724856e-11*pow(XCR, 2)*pow(XNB, 2)*phi_del + 1.0119248343724856e-11*pow(XCR, 2)*pow(XNB, 2)*phi_lav - 1.2880322715205165e-11*pow(XCR, 2)*pow(XNB, 2) - 1.7646996655488141e-11*pow(XCR, 2)*XNB*phi_del - 1.7646996655488141e-11*pow(XCR, 2)*XNB*phi_lav + 2.5563893615445791e-11*pow(XCR, 2)*XNB - 8.0618168522379683e-13*pow(XCR, 2)*phi_del - 8.0618168522379683e-13*pow(XCR, 2)*phi_lav - 3.6266215489870243e-12*pow(XCR, 2) - 8.065202768558045e-12*XCR*pow(XNB, 2)*phi_del - 8.065202768558045e-12*XCR*pow(XNB, 2)*phi_lav + 1.0265823200891526e-11*XCR*pow(XNB, 2) + 7.7964755401576761e-12*XCR*XNB*phi_del + 7.7964755401576761e-12*XCR*XNB*phi_lav - 1.1474697050562422e-11*XCR*XNB + 2.6872722840973019e-13*XCR*phi_del + 2.6872722840973019e-13*XCR*phi_lav + 1.2088738496606712e-12*XCR + 3.4361088799843846e-25*XNB*phi_del + 3.4361088799843846e-25*XNB*phi_lav - 4.3736639081722336e-25*XNB + 1.1448887800131042e-26*phi_del + 1.1448887800131042e-26*phi_lav + 5.1503009766791801e-26;
	return D_CrNb_result;

}

double D_NbCr(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbCr_result;
	D_NbCr_result = -2.7980476658284383e-12*pow(XCR, 3)*XNB*phi_del - 2.7980476658284383e-12*pow(XCR, 3)*XNB*phi_lav - 4.1379136027220271e-13*pow(XCR, 3)*XNB - 3.1352143359064083e-12*pow(XCR, 2)*pow(XNB, 2)*phi_del - 3.1352143359064083e-12*pow(XCR, 2)*pow(XNB, 2)*phi_lav - 1.9305411621056689e-12*pow(XCR, 2)*pow(XNB, 2) + 5.0281364050823661e-12*pow(XCR, 2)*XNB*phi_del + 5.0281364050823661e-12*pow(XCR, 2)*XNB*phi_lav + 7.4358969223535282e-13*pow(XCR, 2)*XNB - 3.3716667007796982e-13*XCR*pow(XNB, 3)*phi_del - 3.3716667007796982e-13*XCR*pow(XNB, 3)*phi_lav - 1.5167498018334665e-12*XCR*pow(XNB, 3) + 2.5672554093268171e-12*XCR*pow(XNB, 2)*phi_del + 2.5672554093268171e-12*XCR*pow(XNB, 2)*phi_lav + 1.8465481337986408e-12*XCR*pow(XNB, 2) - 2.2300887392502495e-12*XCR*XNB*phi_del - 2.2300887392502495e-12*XCR*XNB*phi_lav - 3.2979833196260607e-13*XCR*XNB - 4.6750799613320131e-25*XCR*phi_del - 4.6750799613320131e-25*XCR*phi_lav - 6.9137767744500761e-26*XCR - 5.099249725448536e-26*pow(XNB, 3)*phi_del - 5.099249725448536e-26*pow(XNB, 3)*phi_lav - 2.2939058622801804e-25*pow(XNB, 3) + 1.6908113649415543e-25*pow(XNB, 2)*phi_del + 1.6908113649415543e-25*pow(XNB, 2)*phi_lav + 7.6061426893702937e-25*pow(XNB, 2) - 2.5533964771926475e-25*XNB*phi_del - 2.5533964771926475e-25*XNB*phi_lav - 7.3529193919553314e-25*XNB + 5.6335035394897672e-26*phi_del + 5.6335035394897672e-26*phi_lav + 2.5342408178048241e-25;
	return D_NbCr_result;

}

double D_NbNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbNb_result;
	D_NbNb_result = -3.3716667007796982e-13*pow(XCR, 3)*XNB*phi_del - 3.3716667007796982e-13*pow(XCR, 3)*XNB*phi_lav - 1.5167498018334665e-12*pow(XCR, 3)*XNB + 9.7820816736468872e-12*pow(XCR, 2)*pow(XNB, 2)*phi_del + 9.7820816736468872e-12*pow(XCR, 2)*pow(XNB, 2)*phi_lav - 1.4397072517038633e-11*pow(XCR, 2)*pow(XNB, 2) + 6.0589389848635398e-13*pow(XCR, 2)*XNB*phi_del + 6.0589389848635398e-13*pow(XCR, 2)*XNB*phi_lav + 2.7256236514978403e-12*pow(XCR, 2)*XNB + 1.0119248343724856e-11*XCR*pow(XNB, 3)*phi_del + 1.0119248343724856e-11*XCR*pow(XNB, 3)*phi_lav - 1.2880322715205165e-11*XCR*pow(XNB, 3) - 9.8505211153363088e-12*XCR*pow(XNB, 2)*phi_del - 9.8505211153363088e-12*XCR*pow(XNB, 2)*phi_lav + 1.4089196564890813e-11*XCR*pow(XNB, 2) - 2.6872722840794088e-13*XCR*XNB*phi_del - 2.6872722840794088e-13*XCR*XNB*phi_lav - 1.2088738496623801e-12*XCR*XNB - 5.6335035394897672e-26*XCR*phi_del - 5.6335035394897672e-26*XCR*phi_lav - 2.5342408178048241e-25*XCR + 1.5304174142287522e-24*pow(XNB, 3)*phi_del + 1.5304174142287522e-24*pow(XNB, 3)*phi_lav - 1.9479974712212804e-24*pow(XNB, 3) - 5.0745644877292771e-24*pow(XNB, 2)*phi_del - 5.0745644877292771e-24*pow(XNB, 2)*phi_lav + 6.4591781939619209e-24*pow(XNB, 2) + 4.8004314317403787e-24*XNB*phi_del + 4.8004314317403787e-24*XNB*phi_lav - 6.1763226878074476e-24*XNB - 1.6907608734922438e-24*phi_del - 1.6907608734922438e-24*phi_lav + 2.1520912369273906e-24;
	return D_NbNb_result;

}
