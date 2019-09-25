/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/


#ifndef PRECIPITATEAGING__PARABOLA625__H
#define PRECIPITATEAGING__PARABOLA625__H

double p(double x);
double pPrime(double x);
double interface_profile(double z);
double kT();
double RT();
double Vm();
double xe_gam_Cr();
double xe_gam_Nb();
double xe_del_Cr();
double xe_del_Nb();
double xe_lav_Cr();
double xe_lav_Nb();
double matrix_min_Cr();
double matrix_max_Cr();
double matrix_min_Nb();
double matrix_max_Nb();
double enrich_min_Cr();
double enrich_max_Cr();
double enrich_min_Nb();
double enrich_max_Nb();
double xr_gam_Cr(double r_del, double r_lav);
double xr_gam_Nb(double r_del, double r_lav);
double xr_del_Cr(double r_del, double r_lav);
double xr_del_Nb(double r_del, double r_lav);
double xr_lav_Cr(double r_del, double r_lav);
double xr_lav_Nb(double r_del, double r_lav);
double inv_fict_det(double pDel, double pGam, double pLav);
double fict_gam_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
double fict_gam_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
double fict_del_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
double fict_del_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
double fict_lav_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
double fict_lav_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
double s_delta();
double s_laves();
double CALPHAD_gam(double XCR, double XNB);
double CALPHAD_del(double XCR, double XNB);
double CALPHAD_lav(double XCR, double XNB);
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
double M_CrCr(double XCR, double XNB);
double M_CrNb(double XCR, double XNB);
double M_NbCr(double XCR, double XNB);
double M_NbNb(double XCR, double XNB);

#endif

