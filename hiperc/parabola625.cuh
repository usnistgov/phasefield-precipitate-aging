/******************************************************************************
 *                      Code generated with sympy 1.1.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/


#ifndef PRECIPITATEAGING__PARABOLA625__H
#define PRECIPITATEAGING__PARABOLA625__H

__device__ double h(double x);
__device__ double hprime(double x);
__device__ double xe_gam_Cr();
__device__ double xe_gam_Nb();
__device__ double xe_del_Cr();
__device__ double xe_del_Nb();
__device__ double xe_lav_Cr();
__device__ double xe_lav_Nb();
__device__ double Vm();
__device__ double xr_gam_Cr(double P_del, double P_lav);
__device__ double xr_gam_Nb(double P_del, double P_lav);
__device__ double xr_del_Cr(double P_del, double P_lav);
__device__ double xr_del_Nb(double P_del, double P_lav);
__device__ double xr_lav_Cr(double P_del, double P_lav);
__device__ double xr_lav_Nb(double P_del, double P_lav);
__device__ double fict_gam_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav);
__device__ double fict_gam_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav);
__device__ double fict_del_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav);
__device__ double fict_del_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav);
__device__ double fict_lav_Cr(double XCR, double XNB, double f_del, double f_gam, double f_lav);
__device__ double fict_lav_Nb(double XCR, double XNB, double f_del, double f_gam, double f_lav);
__device__ double r_delta();
__device__ double r_laves();
__device__ double s_delta();
__device__ double s_laves();
__device__ double g_gam(double XCR, double XNB);
__device__ double g_del(double XCR, double XNB);
__device__ double g_lav(double XCR, double XNB);
__device__ double dg_gam_dxCr(double XCR, double XNB);
__device__ double dg_gam_dxNb(double XCR, double XNB);
__device__ double dg_del_dxCr(double XCR, double XNB);
__device__ double dg_del_dxNb(double XCR, double XNB);
__device__ double dg_lav_dxCr(double XCR, double XNB);
__device__ double dg_lav_dxNb(double XCR, double XNB);
__device__ double d2g_gam_dxCrCr();
__device__ double d2g_gam_dxCrNb();
__device__ double d2g_gam_dxNbCr();
__device__ double d2g_gam_dxNbNb();
__device__ double d2g_del_dxCrCr();
__device__ double d2g_del_dxCrNb();
__device__ double d2g_del_dxNbCr();
__device__ double d2g_del_dxNbNb();
__device__ double d2g_lav_dxCrCr();
__device__ double d2g_lav_dxCrNb();
__device__ double d2g_lav_dxNbCr();
__device__ double d2g_lav_dxNbNb();

#endif

