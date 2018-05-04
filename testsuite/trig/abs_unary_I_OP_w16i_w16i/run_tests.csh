#!/bin/csh

testshade --batched -g 200 200 test_abs_w16i_w16i -od uint8 -o res wiwi_ref.tif -o res_m m_wiwi_ref.tif

testshade  -g 200 200 test_abs_w16i_w16i -od uint8 -o res wiwi_out.tif -o res_m m_wiwi_out.tif


idiff wiwi_ref.tif wiwi_out.tif
idiff m_wiwi_ref.tif m_wiwi_out.tif
