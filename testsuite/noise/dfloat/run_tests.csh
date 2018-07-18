rm *.tif *.oso

oslc test_dfloat_1d_u_float.osl
oslc test_dfloat_1d_v_float.osl
oslc test_dfloat_2d_u_float_u_float.osl
oslc test_dfloat_2d_u_float_v_float.osl
oslc test_dfloat_2d_v_float_u_float.osl
oslc test_dfloat_2d_v_float_v_float.osl
oslc test_dfloat_3d_u_point.osl
oslc test_dfloat_3d_v_point.osl
oslc test_dfloat_4d_u_point_u_float.osl
oslc test_dfloat_4d_v_point_u_float.osl
oslc test_dfloat_4d_u_point_v_float.osl
oslc test_dfloat_4d_v_point_v_float.osl

./run_tests_for.csh perlin
./run_tests_for.csh snoise
./run_tests_for.csh uperlin 
./run_tests_for.csh noise
./run_tests_for.csh cell
./run_tests_for.csh simplex
./run_tests_for.csh usimplex
./run_tests_for.csh gabor

