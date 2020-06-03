/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                  This file is part of 'PrecipitateAging'                   *
 ******************************************************************************/


#ifndef D_PRECIPITATEAGING__PARABOLA625__H
#define D_PRECIPITATEAGING__PARABOLA625__H

__device__ double d_p(double x);
__device__ double d_pPrime(double x);
__device__ double d_interface_profile(double z);
__device__ double d_kT();
__device__ double d_RT();
__device__ double d_Vm();
__device__ double d_xe_gam_Cr();
__device__ double d_xe_gam_Nb();
__device__ double d_xe_del_Cr();
__device__ double d_xe_del_Nb();
__device__ double d_xe_lav_Cr();
__device__ double d_xe_lav_Nb();
__device__ double d_matrix_min_Cr();
__device__ double d_matrix_max_Cr();
__device__ double d_matrix_min_Nb();
__device__ double d_matrix_max_Nb();
__device__ double d_enrich_min_Cr();
__device__ double d_enrich_max_Cr();
__device__ double d_enrich_min_Nb();
__device__ double d_enrich_max_Nb();
__device__ double d_xr_gam_Cr(double r_del, double r_lav);
__device__ double d_xr_gam_Nb(double r_del, double r_lav);
__device__ double d_xr_del_Cr(double r_del, double r_lav);
__device__ double d_xr_del_Nb(double r_del, double r_lav);
__device__ double d_xr_lav_Cr(double r_del, double r_lav);
__device__ double d_xr_lav_Nb(double r_del, double r_lav);
__device__ double d_inv_fict_det(double pDel, double pGam, double pLav);
__device__ double d_fict_gam_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
__device__ double d_fict_gam_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
__device__ double d_fict_del_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
__device__ double d_fict_del_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
__device__ double d_fict_lav_Cr(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
__device__ double d_fict_lav_Nb(double INV_DET, double XCR, double XNB, double pDel, double pGam, double pLav);
__device__ double d_s_delta();
__device__ double d_s_laves();
__device__ double d_GCAL_gam(double XCR, double XNB);
__device__ double d_GCAL_del(double XCR, double XNB);
__device__ double d_GCAL_lav(double XCR, double XNB);
__device__ double d_g_gam(double XCR, double XNB);
__device__ double d_g_del(double XCR, double XNB);
__device__ double d_g_lav(double XCR, double XNB);
__device__ double d_dGCAL_gam_dxCr(double XCR, double XNB);
__device__ double d_dGCAL_gam_dxNb(double XCR, double XNB);
__device__ double d_dGCAL_del_dxCr(double XCR, double XNB);
__device__ double d_dGCAL_del_dxNb(double XCR, double XNB);
__device__ double d_dGCAL_lav_dxCr(double XCR, double XNB);
__device__ double d_dGCAL_lav_dxNb(double XCR, double XNB);
__device__ double d_dg_gam_dxCr(double XCR, double XNB);
__device__ double d_dg_gam_dxNb(double XCR, double XNB);
__device__ double d_dg_del_dxCr(double XCR, double XNB);
__device__ double d_dg_del_dxNb(double XCR, double XNB);
__device__ double d_dg_lav_dxCr(double XCR, double XNB);
__device__ double d_dg_lav_dxNb(double XCR, double XNB);
__device__ double d_d2GCAL_gam_dxCrCr(double XCR, double XNB);
__device__ double d_d2GCAL_gam_dxCrNb(double XCR, double XNB);
__device__ double d_d2GCAL_gam_dxNbCr(double XCR, double XNB);
__device__ double d_d2GCAL_gam_dxNbNb(double XCR, double XNB);
__device__ double d_d2GCAL_del_dxCrCr(double XCR);
__device__ double d_d2GCAL_del_dxCrNb(double XNB);
__device__ double d_d2GCAL_del_dxNbCr(double XNB);
__device__ double d_d2GCAL_del_dxNbNb(double XCR, double XNB);
__device__ double d_d2GCAL_lav_dxCrCr(double XCR, double XNB);
__device__ double d_d2GCAL_lav_dxCrNb(double XCR, double XNB);
__device__ double d_d2GCAL_lav_dxNbCr(double XCR, double XNB);
__device__ double d_d2GCAL_lav_dxNbNb(double XCR, double XNB);
__device__ double d_d2g_gam_dxCrCr();
__device__ double d_d2g_gam_dxCrNb();
__device__ double d_d2g_gam_dxNbCr();
__device__ double d_d2g_gam_dxNbNb();
__device__ double d_d2g_del_dxCrCr();
__device__ double d_d2g_del_dxCrNb();
__device__ double d_d2g_del_dxNbCr();
__device__ double d_d2g_del_dxNbNb();
__device__ double d_d2g_lav_dxCrCr();
__device__ double d_d2g_lav_dxCrNb();
__device__ double d_d2g_lav_dxNbCr();
__device__ double d_d2g_lav_dxNbNb();
__device__ double d_mu_Cr(double XCR, double XNB);
__device__ double d_mu_Nb(double XCR, double XNB);
__device__ double d_mu_Ni(double XCR, double XNB);
__device__ double d_D_CrCr(double XCR, double XNB);
__device__ double d_D_CrNb(double XCR, double XNB);
__device__ double d_D_NbCr(double XCR, double XNB);
__device__ double d_D_NbNb(double XCR, double XNB);

#endif

