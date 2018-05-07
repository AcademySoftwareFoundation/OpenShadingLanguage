#!/bin/csh

oslc test_step_w16f_w16f_w16f.osl  
oslc test_step_w16v_w16v_w16v.osl
testshade --batched -g 200 200 test_step_w16fw16fw16f -od uint8 -o res wfwfwf_out.tif -o res_m wfwfwf_m_out.tif
testshade -g 200 200 test_step_w16fw16fw16f -od uint8 -o res wfwfwf_ref.tif -o res_m wfwfwf_m_ref.tif

idiff wfwfwf_ref.tif wfwfwf_out.tif
idiff wfwfwf_m_ref.tif wfwfwf_m_out.tif


testshade --batched -g 200 200 test_step_w16vw16vw16v -od uint8 -o res wvwvwv_out.tif -o res_m wvwvwv_m_out.tif
testshade -g 200 200 test_step_w16vw16vw16v -od uint8 -o res wvwvwv_ref.tif -o res_m wvwvwv_m_ref.tif

idiff wvwvwv_ref.tif wvwvwv_out.tif
idiff wvwvwv_m_ref.tif wvwvwv_m_out.tif

