/******************************************************************************
 *                      Code generated with sympy 1.1.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/


#ifndef PRECIPITATEAGING__PARABOLA625__H
#define PRECIPITATEAGING__PARABOLA625__H

double xe_gam_Cr();
double xe_gam_Nb();
double xe_del_Cr();
double xe_del_Nb();
double xe_lav_Cr();
double xe_lav_Nb();
double Vm();
double xr_gam_Cr(double P_del, double P_lav);
double xr_gam_Nb(double P_del, double P_lav);
double xr_del_Cr(double P_del, double P_lav);
double xr_del_Nb(double P_del, double P_lav);
double xr_lav_Cr(double P_del, double P_lav);
double xr_lav_Nb(double P_del, double P_lav);
double fict_gam_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav);
double fict_gam_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav);
double fict_del_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav);
double fict_del_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav);
double fict_lav_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav);
double fict_lav_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav);
double r_delta();
double r_laves();
double s_delta();
double s_laves();
double g_gam(double XCR, double XNB);
double g_del(double XCR, double XNB);
double g_lav(double XCR, double XNB);
double dg_gam_dxCr(double XCR, double XNB);
double dg_gam_dxNb(double XCR, double XNB);
double dg_del_dxCr(double XCR, double XNB);
double dg_del_dxNb(double XCR, double XNB);
double dg_lav_dxCr(double XCR, double XNB);
double dg_lav_dxNb(double XCR, double XNB);
double d2g_gam_dxCrCr();
double d2g_gam_dxCrNb();
double d2g_gam_dxNbCr();
double d2g_gam_dxNbNb();
double d2g_del_dxCrCr();
double d2g_del_dxCrNb();
double d2g_del_dxNbCr();
double d2g_del_dxNbNb();
double d2g_lav_dxCrCr();
double d2g_lav_dxCrNb();
double d2g_lav_dxNbCr();
double d2g_lav_dxNbNb();

#endif

