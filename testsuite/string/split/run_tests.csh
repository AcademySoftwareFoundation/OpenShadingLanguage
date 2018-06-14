#!/bin/csh

oslc split_u_string_u_sep_u_maxplit.osl 
oslc split_u_string_u_sep_v_maxsplit.osl 

oslc split_u_string_v_sep_u_maxplit.osl 
oslc split_u_string_v_sep_v_maxsplit.osl 


oslc split_v_string_u_sep_u_maxsplit.osl 
oslc split_v_string_u_sep_v_maxsplit.osl 

oslc split_v_string_v_sep_u_maxsplit.osl 
oslc split_v_string_v_sep_v_maxsplit.osl 



##############################################
#Uniform string, uniform sep, uniform maxsplit
##############################################
testshade --batched -g 200 200 test_split_u_str_u_sep_u_max -od uint8 -o res split_uuu_out.tif -o calres msplit_uuu_out.tif
testshade -g 200 200 test_split_u_str_u_sep_u_max -od uint8 -o res split_uuu_ref.tif -o calres msplit_uuu_ref.tif
idiff split_uuu_ref.tif split_uuu_out.tif
idiff msplit_uuu_ref.tif msplit_uuu_out.tif

##############################################
#Uniform string, uniform sep, varying maxsplit  
##############################################
testshade --batched -g 200 200 test_split_u_str_u_sep_v_max -od uint8 -o res split_uuv_out.tif -o calres msplit_uuv_out.tif
testshade -g 200 200 test_split_u_str_u_sep_v_max -od uint8 -o res split_uuv_ref.tif -o calres msplit_uuv_ref.tif
idiff split_uuv_ref.tif split_uuv_out.tif
idiff msplit_uuv_ref.tif msplit_uuv_out.tif


##############################################
#Uniform string, varying sep, uniform maxsplit 
##############################################
testshade --batched -g 200 200 test_split_u_str_v_sep_u_max -od uint8 -o res split_uvu_out.tif -o calres msplit_uvu_out.tif
testshade -g 200 200 test_split_u_str_v_sep_u_max -od uint8 -o res split_uvu_ref.tif -o calres msplit_uvu_ref.tif
idiff split_uvu_ref.tif split_uvu_out.tif
idiff msplit_uvu_ref.tif msplit_uvu_out.tif


##############################################
#Uniform string, varying sep, varying maxsplit 
##############################################
testshade --batched -g 200 200 test_split_u_str_v_sep_v_max -od uint8 -o res split_uvv_out.tif -o calres msplit_uvv_out.tif
testshade -g 200 200 test_split_u_str_v_sep_v_max -od uint8 -o res split_uvv_ref.tif -o calres msplit_uvv_ref.tif
idiff split_uvv_ref.tif split_uvv_out.tif
idiff msplit_uvv_ref.tif msplit_uvv_out.tif


##############################################
#Varying string, uniform sep, uniform maxsplit 
##############################################
testshade --batched -g 200 200 test_split_v_str_u_sep_u_max -od uint8 -o res split_vuu_out.tif -o calres msplit_vuu_out.tif
testshade -g 200 200 test_split_v_str_u_sep_u_max -od uint8 -o res split_vuu_ref.tif -o calres msplit_vuu_ref.tif
idiff split_vuu_ref.tif split_vuu_out.tif
idiff msplit_vuu_ref.tif msplit_vuu_out.tif




##############################################
#Varying string, uniform sep, varying maxsplit 
##############################################
testshade --batched -g 200 200 test_split_v_str_u_sep_v_max -od uint8 -o res split_vuv_out.tif -o calres msplit_vuv_out.tif
testshade -g 200 200 test_split_v_str_u_sep_v_max -od uint8 -o res split_vuv_ref.tif -o calres msplit_vuv_ref.tif
idiff split_vuv_ref.tif split_vuv_out.tif
idiff msplit_vuv_ref.tif msplit_vuv_out.tif



##############################################
#varying string, varying sep, uniform maxsplit 
##############################################
testshade --batched -g 200 200 test_split_v_str_v_sep_u_max -od uint8 -o res split_vvu_out.tif -o calres msplit_vvu_out.tif
testshade -g 200 200 test_split_v_str_v_sep_u_max -od uint8 -o res split_vvu_ref.tif -o calres msplit_vvu_ref.tif
idiff split_vvu_ref.tif split_vvu_out.tif
idiff msplit_vvu_ref.tif msplit_vvu_out.tif


##############################################
#varying string, varying sep, varying maxsplit 
##############################################
testshade --batched -g 200 200 test_split_v_str_v_sep_v_max -od uint8 -o res split_vvv_out.tif -o calres msplit_vvv_out.tif
testshade -g 200 200 test_split_v_str_v_sep_v_max  -od uint8 -o res split_vvv_ref.tif -o calres msplit_vvv_ref.tif
idiff split_vvv_ref.tif split_vvv_out.tif
idiff msplit_vvv_ref.tif msplit_vvv_out.tif
