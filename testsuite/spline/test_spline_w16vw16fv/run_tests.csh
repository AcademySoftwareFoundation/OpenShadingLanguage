#!/bin/csh

set ITER_NUM=2
oslc test_spline.osl

echo "\n"
echo "*******************"
echo "Catmull-Rom spline..."
echo "*******************"


testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 catmullrom_out.tif  \
                                                                              -o mFspline1 mcatmullrom_out.tif  \
                                                                              -param splinename catmull-rom test_spline
                                                                              
testshade -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 catmullrom_ref.tif  \
                                                                              -o mFspline1 mcatmullrom_ref.tif \
                                                                             -param splinename catmull-rom test_spline
idiff catmullrom_ref.tif catmullrom_out.tif
idiff mcatmullrom_ref.tif mcatmullrom_out.tif


echo "\n"
echo "*******************"
echo "Bezier spline..."
echo "*******************"


testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 bezier_out.tif  \
                                                                              -o mFspline1 mbezier_out.tif \
                                                                              -param splinename bezier test_spline
                                                                              
testshade -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 bezier_ref.tif  \
                                                                              -o mFspline1 mbezier_ref.tif \
                                                                             -param splinename bezier test_spline
idiff bezier_ref.tif bezier_out.tif
idiff mbezier_ref.tif mbezier_out.tif



echo "\n"
echo "*******************"
echo "Bspline spline..."
echo "*******************"


testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 bspline_out.tif  \
                                                                              -o mFspline1 mbspline_out.tif \
                                                                              -param splinename bspline test_spline
                                                                              
testshade -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 bspline_ref.tif  \
                                                                              -o mFspline1 mbspline_ref.tif \
                                                                             -param splinename bspline test_spline
idiff bspline_ref.tif bspline_out.tif
idiff mbspline_ref.tif mbspline_out.tif



echo "\n"
echo "*******************"
echo "Hermite spline..."
echo "*******************"


testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 hermite_out.tif  \
                                                                              -o mFspline1 mhermite_out.tif  \
                                                                              -param splinename hermite test_spline
                                                                              
testshade -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 hermite_ref.tif  \
                                                                              -o mFspline1 mhermite_ref.tif \
                                                                             -param splinename hermite test_spline
idiff hermite_ref.tif hermite_out.tif
idiff mhermite_ref.tif mhermite_out.tif

echo "\n"
echo "*******************"
echo "Linear spline..."
echo "*******************"


testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 linear_out.tif  \
                                                                              -o mFspline1 mlinear_out.tif \
                                                                              -param splinename linear test_spline
                                                                              
testshade -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 linear_ref.tif  \
                                                                              -o mFspline1 mlinear_ref.tif \
                                                                             -param splinename linear test_spline
idiff linear_ref.tif linear_out.tif
idiff mlinear_ref.tif mlinear_out.tif




echo "\n"
echo "*******************"
echo "Constant spline..."
echo "*******************"


testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 constant_out.tif   \
                                                                              -o mFspline1 mconstant_out.tif  \
                                                                              -param splinename constant test_spline
                                                                              
testshade -g 1024 1024 -t 1 --iters $ITER_NUM  -od uint8 -o Fspline1 constant_ref.tif \
                                                                              -o mFspline1 mconstant_ref.tif \
                                                                             -param splinename constant test_spline
idiff constant_ref.tif constant_out.tif
idiff mconstant_ref.tif mconstant_out.tif







