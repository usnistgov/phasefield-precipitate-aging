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
	D_CrCr_result = -9.508313026280806e-12*(0.29427382734399732*pow(XCR, 4)*phi_del + 0.10615002959815199*pow(XCR, 4)*phi_lav + 0.043518903840091057*pow(XCR, 4) + 0.32973402613489194*pow(XCR, 3)*XNB*phi_del + 0.56846837325392785*pow(XCR, 3)*XNB*phi_lav + 0.20303719037958556*pow(XCR, 3)*XNB - 0.7633557208591466*pow(XCR, 3)*phi_del - 0.27535657212353853*pow(XCR, 3)*phi_lav - 0.11288942856959956*pow(XCR, 3) + 0.035460198790894445*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.46231834365577595*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.15951828653949415*pow(XCR, 2)*pow(XNB, 2) - 0.5328044378619704*pow(XCR, 2)*XNB*phi_del - 1.0*pow(XCR, 2)*XNB*phi_lav - 0.35602743684035476*pow(XCR, 2)*XNB + 0.70362284027237654*pow(XCR, 2)*phi_del + 0.25380981378795431*pow(XCR, 2)*phi_lav + 0.10405578709421082*pow(XCR, 2) - 0.028262345556455445*XCR*pow(XNB, 2)*phi_del - 0.36847511381810993*XCR*pow(XNB, 2)*phi_lav - 0.12713862557111547*XCR*pow(XNB, 2) + 0.26280329231377753*XCR*XNB*phi_del + 0.45307838508103948*XCR*XNB*phi_lav + 0.16182388793594457*XCR*XNB - 0.23454094675754641*XCR*phi_del - 0.084603271262607652*XCR*phi_lav - 0.034685262364724107*XCR + 1.2040924366379736e-15*XNB*phi_del + 1.5698558944848818e-14*XNB*phi_lav + 5.4166295981672465e-15*XNB - 9.9924112635375631e-15*phi_del - 3.6044481459830942e-15*phi_lav - 1.4777351721469961e-15);
	return D_CrCr_result;

}

double D_CrNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_CrNb_result;
	D_CrNb_result = -3.3718817146638498e-11*(0.0099993623326606346*pow(XCR, 4)*phi_del + 0.13036837888333264*pow(XCR, 4)*phi_lav + 0.044982295649261074*pow(XCR, 4) - 0.29010749787295209*pow(XCR, 3)*XNB*phi_del + 0.57120398473212541*pow(XCR, 3)*XNB*phi_lav + 0.426974423640893*pow(XCR, 3)*XNB - 0.025938665733453613*pow(XCR, 3)*phi_del - 0.33817974482450935*pow(XCR, 3)*phi_lav - 0.11668551373106464*pow(XCR, 3) - 0.30010686020561278*pow(XCR, 2)*pow(XNB, 2)*phi_del + 0.44083560584879261*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.3819921279916319*pow(XCR, 2)*pow(XNB, 2) + 0.52335752404195512*pow(XCR, 2)*XNB*phi_del - 1.0*pow(XCR, 2)*XNB*phi_lav - 0.75814918133907061*pow(XCR, 2)*XNB + 0.023908955101177572*pow(XCR, 2)*phi_del + 0.31171704891161106*pow(XCR, 2)*phi_lav + 0.10755482712265217*pow(XCR, 2) + 0.23918996723650146*XCR*pow(XNB, 2)*phi_del - 0.35135302820939618*XCR*pow(XNB, 2)*phi_lav - 0.3044538352649464*XCR*pow(XNB, 2) - 0.23122031553633332*XCR*XNB*phi_del + 0.45525871118018252*XCR*XNB*phi_lav + 0.340305444306084*XCR*XNB - 0.0079696517004458414*XCR*phi_del - 0.10390568297050479*XCR*phi_lav - 0.035851609040834537*XCR - 1.0190478702266512e-14*XNB*phi_del + 1.4969087509445936e-14*XNB*phi_lav + 1.2970988540765742e-14*XNB - 3.3954001857008808e-16*phi_del - 4.4268104619448014e-15*phi_lav - 1.527426348997128e-15);
	return D_CrNb_result;

}

double D_NbCr(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbCr_result;
	D_NbCr_result = -5.4051752384389819e-12*(0.51766086063779804*pow(XCR, 3)*XNB*phi_del + 0.18672987732025695*pow(XCR, 3)*XNB*phi_lav + 0.076554661415880909*pow(XCR, 3)*XNB + 0.58003935073377211*pow(XCR, 2)*pow(XNB, 2)*phi_del + 1.0*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.35716532340646384*pow(XCR, 2)*pow(XNB, 2) - 0.93024484559255394*pow(XCR, 2)*XNB*phi_del - 0.33555657594294797*pow(XCR, 2)*XNB*phi_lav - 0.1375699509143207*pow(XCR, 2)*XNB + 0.062378490095973745*XCR*pow(XNB, 3)*phi_del + 0.81327012267974319*XCR*pow(XNB, 3)*phi_lav + 0.28061066199058232*XCR*pow(XNB, 3) - 0.47496247504979006*XCR*pow(XNB, 2)*phi_del - 0.96209682130359131*XCR*pow(XNB, 2)*phi_lav - 0.34162595148939612*XCR*pow(XNB, 2) + 0.41258398495407883*XCR*XNB*phi_del + 0.14882669862244519*XCR*XNB*phi_lav + 0.061015289498338732*XCR*XNB + 8.6492662219073217e-14*XCR*phi_del + 3.1199508082127346e-14*XCR*phi_lav + 1.2791031686230364e-14*XCR + 9.4340137007680134e-15*pow(XNB, 3)*phi_del + 1.2299755040529861e-13*pow(XNB, 3)*phi_lav + 4.243906554531361e-14*pow(XNB, 3) - 3.1281342238773787e-14*pow(XNB, 2)*phi_del - 4.0783579405293329e-13*pow(XNB, 2)*phi_lav - 1.4071963172033909e-13*pow(XNB, 2) + 4.7239846342707486e-14*XNB*phi_del + 3.930648583421063e-13*XNB*phi_lav + 1.3603480123391633e-13*XNB - 1.0422425344190555e-14*phi_del - 1.3588413450291879e-13*phi_lav - 4.6885451553587647e-14);
	return D_NbCr_result;

}

double D_NbNb(double XCR, double XNB, double phi_del, double phi_lav)
{

	double D_NbNb_result;
	D_NbNb_result = -1.9260322714613821e-11*(0.017505764315264689*pow(XCR, 3)*XNB*phi_del + 0.22823436524951912*pow(XCR, 3)*XNB*phi_lav + 0.078749968227823541*pow(XCR, 3)*XNB - 0.50788773472755511*pow(XCR, 2)*pow(XNB, 2)*phi_del + 1.0*pow(XCR, 2)*pow(XNB, 2)*phi_lav + 0.74749902846201088*pow(XCR, 2)*pow(XNB, 2) - 0.031458138446799261*pow(XCR, 2)*XNB*phi_del - 0.41014080453922369*pow(XCR, 2)*XNB*phi_lav - 0.14151495236524597*pow(XCR, 2)*XNB - 0.52539349904281984*XCR*pow(XNB, 3)*phi_del + 0.77176563475048077*XCR*pow(XNB, 3)*phi_lav + 0.66874906023418734*XCR*pow(XNB, 3) + 0.51144112491231519*XCR*pow(XNB, 2)*phi_del - 0.95367207404117948*XCR*pow(XNB, 2)*phi_lav - 0.73151404437271428*XCR*pow(XNB, 2) + 0.013952374131511456*XCR*XNB*phi_del + 0.18190643928940456*XCR*XNB*phi_lav + 0.062764984137318958*XCR*XNB + 2.9249268680297506e-15*XCR*phi_del + 3.8134229108975523e-14*XCR*phi_lav + 1.3157831544961407e-14*XCR - 7.9459593533577906e-14*pow(XNB, 3)*phi_del + 1.1672048426975112e-13*pow(XNB, 3)*phi_lav + 1.0114043778421387e-13*pow(XNB, 3) + 2.6347245385867486e-13*pow(XNB, 2)*phi_del - 3.8702227180571711e-13*pow(XNB, 2)*phi_lav - 3.3536188825439576e-13*pow(XNB, 2) - 2.4923940802393918e-13*XNB*phi_del + 3.7473807868547952e-13*XNB*phi_lav + 3.2067597097535364e-13*XNB + 8.7784659610577274e-14*phi_del - 1.2894941347607053e-13*phi_lav - 1.1173702895925442e-13);
	return D_NbNb_result;

}
