#!/bin/csh

./run_test_dir.csh test_spline_w16dvw16fdv
./run_test_dir.csh test_spline_w16dvw16fw16dv
./run_test_dir.csh test_spline_w16dfw16fw16df
./run_test_dir.csh test_spline_w16dvdfw16v
./run_test_dir.csh test_spline_w16dvw16dfw16v
./run_test_dir.csh test_spline_w16dvdfw16dv
./run_test_dir.csh test_spline_w16dvw16dfdv
./run_test_dir.csh test_spline_w16vfw16v
./run_test_dir.csh test_spline_w16vw16fw16v
./run_test_dir.csh test_spline_w16dfdfw16df
./run_test_dir.csh test_spline_w16dfw16dfdf
./run_test_dir.csh test_spline_w16ffw16f
./run_test_dir.csh test_spline_w16fw16ff
./run_test_dir.csh test_spline_w16fff
./run_test_dir.csh test_spline_w16fw16fw16f
./run_test_dir.csh test_spline_w16vw16fv
./run_test_dir.csh test_spline_w16dvfw16dv
./run_test_dir.csh test_spline_w16dvw16dfv
./run_test_dir.csh test_spline_w16dfw16dfw16df
./run_test_dir.csh test_spline_w16dvw16dfw16dv
./run_test_dir.csh test_spline_w16dffw16df
./run_test_dir.csh test_spline_w16dfw16dff

#New wide output, uniform operand tests

./run_test_dir.csh trial/test_spline_w16df_df_df
./run_test_dir.csh trial/test_spline_w16df_df_f
./run_test_dir.csh trial/test_spline_w16df_f_df

./run_test_dir.csh trial/test_spline_w16dv_df_dv
./run_test_dir.csh trial/test_spline_w16dv_df_v
./run_test_dir.csh trial/test_spline_w16dv_f_dv

./run_test_dir.csh trial/test_spline_w16v_f_v


