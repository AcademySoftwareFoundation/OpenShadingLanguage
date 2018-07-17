rm *.tif *.oso
rm run_log.txt
oslc benchmark.osl

./run_benchmark.csh null 1000000
./run_benchmark.csh sin 10000
./run_benchmark.csh cos 10000
./run_benchmark.csh tan 10000
./run_benchmark.csh asin 100000
./run_benchmark.csh acos 100000
./run_benchmark.csh atan 10000
./run_benchmark.csh sinh 10000
./run_benchmark.csh cosh 10000
./run_benchmark.csh tanh 10000
./run_benchmark.csh atan2 10000
./run_benchmark.csh sincos 10000
./run_benchmark.csh log 10000
./run_benchmark.csh log2 10000
./run_benchmark.csh log10 10000
./run_benchmark.csh logb 100000
./run_benchmark.csh exp 10000
./run_benchmark.csh exp2 10000
./run_benchmark.csh expm1 10000
./run_benchmark.csh pow 10000
./run_benchmark.csh erf 10000
./run_benchmark.csh erfc 10000
./run_benchmark.csh radians 500000
./run_benchmark.csh degrees 500000
./run_benchmark.csh sqrt 100000
./run_benchmark.csh inversesqrt 100000
./run_benchmark.csh hypot 100000
./run_benchmark.csh abs 100000
./run_benchmark.csh fabs 100000
./run_benchmark.csh sign 100000
./run_benchmark.csh floor 100000
./run_benchmark.csh ceil 100000
./run_benchmark.csh round 100000
./run_benchmark.csh trunc 100000
./run_benchmark.csh mod 100000
#./run_benchmark.csh fmod 100000
./run_benchmark.csh min 600000
./run_benchmark.csh max 600000
./run_benchmark.csh clamp 600000
./run_benchmark.csh mix 400000
./run_benchmark.csh isnan 100000
./run_benchmark.csh isfinite 100000
#./run_benchmark.csh isinf 100000
./run_benchmark.csh select 300000
./run_benchmark.csh dot 300000
./run_benchmark.csh cross 300000
./run_benchmark.csh length 200000
./run_benchmark.csh distance 200000
./run_benchmark.csh normalize 100000
./run_benchmark.csh reflect 300000
./run_benchmark.csh fresnel 50000
./run_benchmark.csh rotate 10000
./run_benchmark.csh transform 20000
./run_benchmark.csh transform_matrix 200000
./run_benchmark.csh matrix_object_camera 20000
./run_benchmark.csh determinant 100000
./run_benchmark.csh transpose 100000
#./run_benchmark.csh step 100000
./run_benchmark.csh linearstep 250000
#./run_benchmark.csh smoothstep 100000
./run_benchmark.csh smooth_linearstep 100000
./run_benchmark.csh noise_perlin 15000
./run_benchmark.csh noise_cell 30000
./run_benchmark.csh noise_simplex 5000
./run_benchmark.csh noise_gabor 25
./run_benchmark.csh pnoise_perlin 10000
./run_benchmark.csh pnoise_cell 20000
./run_benchmark.csh pnoise_gabor 25
./run_benchmark.csh spline_bezier 30000
./run_benchmark.csh spline_bspline 30000
./run_benchmark.csh spline_catmull-rom 30000
./run_benchmark.csh spline_hermite 30000
./run_benchmark.csh spline_linear 30000
./run_benchmark.csh spline_constant 90000









