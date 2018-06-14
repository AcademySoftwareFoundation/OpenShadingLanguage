#!/bin/csh


oslc wavelength_u_lambda_masked.osl
oslc wavelength_v_lambda_masked.osl

oslc wavelength_u_lambda.osl
oslc wavelength_v_lambda.osl

########################################
#Varying lambda 
########################################
testshade --batched -g 200 200 wavelength_v_lambda -od uint8 -o Cout v_lambda_batched_out.tif
testshade -g 200 200 wavelength_v_lambda -od uint8 -o Cout v_lambda_batched_ref.tif
idiff v_lambda_batched_out.tif v_lambda_batched_ref.tif


########################################
#Uniform lambda 
########################################
testshade --batched -g 200 200 wavelength_u_lambda -od uint8 -o Cout u_lambda_batched_out.tif
testshade -g 200 200 wavelength_u_lambda -od uint8 -o Cout u_lambda_batched_ref.tif
idiff u_lambda_batched_out.tif u_lambda_batched_ref.tif

########################################
#Uniform lambda--masked 
########################################
testshade --batched -g 200 200 wavelength_u_lambda_masked -od uint8 -o Cout u_lambda_masked_out.tif
testshade -g 200 200 wavelength_u_lambda_masked -od uint8 -o Cout u_lambda_masked_ref.tif
idiff u_lambda_masked_ref.tif u_lambda_masked_out.tif


########################################
#Varying lambda--masked 
########################################
testshade --batched -g 200 200 wavelength_v_lambda_masked -od uint8 -o Cout v_lambda_masked_out.tif
testshade -g 200 200 wavelength_v_lambda_masked -od uint8 -o Cout v_lambda_masked_ref.tif
idiff v_lambda_masked_ref.tif v_lambda_masked_out.tif









