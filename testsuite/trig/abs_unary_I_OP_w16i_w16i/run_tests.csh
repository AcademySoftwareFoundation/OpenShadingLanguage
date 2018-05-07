#!/bin/csh

oslc test_abs_w16i_w16i.osl

testshade --batched -g 200 200 test_abs_w16i_w16i -od uint8 -o res wiwi_ref.tif -o res_m m_wiwi_ref.tif

testshade  -g 200 200 test_abs_w16i_w16i -od uint8 -o res wiwi_out.tif -o res_m m_wiwi_out.tif


idiff -fail 0.004   wiwi_ref.tif wiwi_out.tif
idiff -fail 0.004   m_wiwi_ref.tif m_wiwi_out.tif
