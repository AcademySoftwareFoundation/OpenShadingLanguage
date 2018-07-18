#!/bin/csh


./exec_test_dir.csh select

./exec_test_dir.csh noise 
./exec_test_dir.csh pnoise

./run_test_dir.csh calculatenormal 


#./run_test_dir.csh spline



./run_test_dir.csh spline/test_spline_w16dvw16fdv
./run_test_dir.csh spline/test_spline_w16dvw16fw16dv
./run_test_dir.csh spline/test_spline_w16dfw16fw16df
./run_test_dir.csh spline/test_spline_w16dvdfw16v
./run_test_dir.csh spline/test_spline_w16dvw16dfw16v
./run_test_dir.csh spline/test_spline_w16dvdfw16dv
./run_test_dir.csh spline/test_spline_w16dvw16dfdv
./run_test_dir.csh spline/test_spline_w16vfw16v
./run_test_dir.csh spline/test_spline_w16vw16fw16v
./run_test_dir.csh spline/test_spline_w16dfdfw16df
./run_test_dir.csh spline/test_spline_w16dfw16dfdf
./run_test_dir.csh spline/test_spline_w16ffw16f
./run_test_dir.csh spline/test_spline_w16fw16ff
./run_test_dir.csh spline/test_spline_w16fff
./run_test_dir.csh spline/test_spline_w16fw16fw16f
./run_test_dir.csh spline/test_spline_w16vw16fv
./run_test_dir.csh spline/test_spline_w16dvfw16dv
./run_test_dir.csh spline/test_spline_w16dvw16dfv
./run_test_dir.csh spline/test_spline_w16dfw16dfw16df
./run_test_dir.csh spline/test_spline_w16dvw16dfw16dv
./run_test_dir.csh spline/test_spline_w16dffw16df
./run_test_dir.csh spline/test_spline_w16dfw16dff


./run_test_dir.csh spline/trial/test_spline_w16df_df_df
./run_test_dir.csh spline/trial/test_spline_w16df_df_f
./run_test_dir.csh spline/trial/test_spline_w16df_f_df

./run_test_dir.csh spline/trial/test_spline_w16dv_df_dv
./run_test_dir.csh spline/trial/test_spline_w16dv_df_v
./run_test_dir.csh spline/trial/test_spline_w16dv_f_dv

./run_test_dir.csh spline/trial/test_spline_w16v_f_v




./run_test_dir.csh trig/pow_vf
./exec_test_dir.csh trace
./exec_test_dir.csh message

./run_test_dir.csh string/split
./run_test_dir.csh blackbody
./run_test_dir.csh wavelength_color
./exec_test_dir.csh trace

./exec_test_dir.csh matrix
./run_test_dir.csh length
./exec_test_dir.csh transform
./run_test_dir.csh normalize
./exec_test_dir.csh blendmath
./exec_test_dir.csh construct_triple
./run_test_dir.csh mxcompref
./run_test_dir.csh arraycopy

#SM: Added line 10 to 46
./run_test_dir.csh trig/abs_unary_I_OP_w16i_w16i
./run_test_dir.csh trig/pow
./run_test_dir.csh trig/atan2
./run_test_dir.csh trig/fmod
./run_test_dir.csh trig/sincos

./run_test_dir.csh trig/acos
./run_test_dir.csh trig/asin
./run_test_dir.csh trig/atan

./run_test_dir.csh trig/cos
./run_test_dir.csh trig/sin
./run_test_dir.csh trig/tan

./run_test_dir.csh trig/cosh
./run_test_dir.csh trig/sinh
./run_test_dir.csh trig/tanh

./run_test_dir.csh trig/log2
./run_test_dir.csh trig/log
./run_test_dir.csh trig/log10


./run_test_dir.csh trig/exp
./run_test_dir.csh trig/exp2
./run_test_dir.csh trig/expm1

./run_test_dir.csh trig/sqrt
./run_test_dir.csh trig/inversesqrt

./run_test_dir.csh trig/erf
./run_test_dir.csh trig/erfc

./run_test_dir.csh blendmath/step

./run_test_dir.csh string

./run_test_dir.csh error-dupes/error-warning 16





./run_test_dir.csh vector/dot_tests
./exec_test_dir.csh vector/distance_tests
./run_test_dir.csh vector/cross_tests

./run_test_dir.csh filterwidth

./run_test_dir.csh compassign

./run_test_dir.csh abs
./run_test_dir.csh neg

./run_test_dir.csh smoothstep
./run_test_dir.csh color

./run_test_dir.csh texture-filename
./run_test_dir.csh texture-width
./run_test_dir.csh texture-firstchannel
./run_test_dir.csh texture-missingcolor
./run_test_dir.csh texture-missingalpha
./run_test_dir.csh texture-subimage
./run_test_dir.csh texture-blur
./run_test_dir.csh texture-interp
./run_test_dir.csh texture-subimagename 10
./run_test_dir.csh texture-wrap

./run_test_dir.csh array 
./run_test_dir.csh array_assign
./run_test_dir.csh exit

./exec_test_dir.csh shaderglobals 
./exec_test_dir.csh breakcont 

./exec_test_dir.csh div 
./exec_test_dir.csh mul
./exec_test_dir.csh add
./exec_test_dir.csh sub
./exec_test_dir.csh max
./exec_test_dir.csh min

./run_test_dir.csh luminance
