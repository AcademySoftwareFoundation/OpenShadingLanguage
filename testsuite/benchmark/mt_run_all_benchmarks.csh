rm *.tif *.oso
rm t40_run_log.txt
oslc benchmark.osl

./mt_run_benchmark.csh null 100000
./mt_run_benchmark.csh sin 1000
./mt_run_benchmark.csh cos 1000
./mt_run_benchmark.csh tan 1000
./mt_run_benchmark.csh asin 3000
./mt_run_benchmark.csh acos 3000
./mt_run_benchmark.csh atan 1000
./mt_run_benchmark.csh sinh 1000
./mt_run_benchmark.csh cosh 1000
./mt_run_benchmark.csh tanh 1000
./mt_run_benchmark.csh atan2 1000
./mt_run_benchmark.csh sincos 1000
./mt_run_benchmark.csh log 1000
./mt_run_benchmark.csh log2 1000
./mt_run_benchmark.csh log10 1000
./mt_run_benchmark.csh logb 2000
./mt_run_benchmark.csh exp 1000
./mt_run_benchmark.csh exp2 1000
./mt_run_benchmark.csh expm1 1000
./mt_run_benchmark.csh pow 1000
./mt_run_benchmark.csh erf 1000
./mt_run_benchmark.csh erfc 1000
./mt_run_benchmark.csh radians 50000
./mt_run_benchmark.csh degrees 50000
./mt_run_benchmark.csh sqrt 10000
./mt_run_benchmark.csh inversesqrt 10000
./mt_run_benchmark.csh hypot 1000
./mt_run_benchmark.csh abs 10000
./mt_run_benchmark.csh fabs 10000
./mt_run_benchmark.csh sign 3000
./mt_run_benchmark.csh floor 10000
./mt_run_benchmark.csh ceil 10000
./mt_run_benchmark.csh round 3000
./mt_run_benchmark.csh trunc 6000
./mt_run_benchmark.csh mod 5000
#./mt_run_benchmark.csh fmod 1000
./mt_run_benchmark.csh min 10000
./mt_run_benchmark.csh max 10000
./mt_run_benchmark.csh clamp 10000
./mt_run_benchmark.csh mix 10000
./mt_run_benchmark.csh isnan 10000
./mt_run_benchmark.csh isfinite 6000
#./mt_run_benchmark.csh isinf 10000
./mt_run_benchmark.csh select 30000
./mt_run_benchmark.csh dot 10000
./mt_run_benchmark.csh cross 15000
./mt_run_benchmark.csh length 2000
./mt_run_benchmark.csh distance 10000
./mt_run_benchmark.csh normalize 3000
./mt_run_benchmark.csh reflect 15000
./mt_run_benchmark.csh fresnel 2500
./mt_run_benchmark.csh rotate 500
./mt_run_benchmark.csh transform 1000
./mt_run_benchmark.csh transform_matrix 10000
./mt_run_benchmark.csh matrix_object_camera 500
./mt_run_benchmark.csh determinant 7000
./mt_run_benchmark.csh transpose 1000
#./mt_run_benchmark.csh step 10000
./mt_run_benchmark.csh linearstep 25000
#./mt_run_benchmark.csh smoothstep 1000
./mt_run_benchmark.csh smooth_linearstep 4000
./mt_run_benchmark.csh noise_perlin 450
./mt_run_benchmark.csh noise_cell 3000
./mt_run_benchmark.csh noise_simplex 200
./mt_run_benchmark.csh noise_gabor 10
./mt_run_benchmark.csh pnoise_perlin 450
./mt_run_benchmark.csh pnoise_cell 2000
./mt_run_benchmark.csh pnoise_gabor 10
./mt_run_benchmark.csh spline_bezier 1000
./mt_run_benchmark.csh spline_bspline 1000
./mt_run_benchmark.csh spline_catmull-rom 1000
./mt_run_benchmark.csh spline_hermite 1000
./mt_run_benchmark.csh spline_linear 1000
./mt_run_benchmark.csh spline_constant 9000









