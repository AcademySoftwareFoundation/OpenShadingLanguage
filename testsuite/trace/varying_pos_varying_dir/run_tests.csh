#!/bin/csh


cd uniform_traceset


oslc trace_v_point_v_dir_u_mindist_v_maxdist_u_traceset.osl  
oslc trace_v_point_v_dir_u_mindist_u_maxdist_u_traceset.osl
oslc trace_v_point_v_dir_v_mindist_u_maxdist_u_traceset.osl  
oslc trace_v_point_v_dir_v_mindist_v_maxdist_u_traceset.osl

###########################################
#Varying position; Varying direction
#Uniform mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_vv_uu_out.tif -o mcout trace_vv_uu_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_vv_uu_ref.tif -o mcout trace_vv_uu_m_ref.tif

idiff trace_vv_uu_ref.tif trace_vv_uu_out.tif
idiff trace_vv_uu_m_ref.tif trace_vv_uu_m_out.tif

###########################################
#Uniform position; Uniform direction
#Varying mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_v_mindist_u_maxdist_u_traceset -od uint8 -o cout trace_vv_vu_out.tif -o mcout trace_vv_vu_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_v_mindist_u_maxdist_u_traceset -od uint8 -o cout trace_vv_vu_ref.tif -o mcout trace_vv_vu_m_ref.tif

idiff trace_vv_vu_ref.tif trace_vv_vu_out.tif
idiff trace_vv_vu_m_ref.tif trace_vv_vu_m_out.tif


###########################################
#Uniform position; Uniform direction
#Uniform mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_u_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_vv_uv_out.tif -o mcout trace_vv_uv_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_u_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_vv_uv_ref.tif -o mcout trace_vv_uv_m_ref.tif

idiff trace_vv_uv_ref.tif trace_vv_uv_out.tif
idiff trace_vv_uv_m_ref.tif trace_vv_uv_m_out.tif



###########################################
#Uniform position; Uniform direction
#Varying mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_vv_vv_out.tif -o mcout trace_vv_vv_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_vv_vv_ref.tif -o mcout trace_vv_vv_m_ref.tif

idiff trace_vv_vv_ref.tif trace_vv_vv_out.tif
idiff trace_vv_vv_m_ref.tif trace_vv_vv_m_out.tif

cd ..
cd varying_traceset

oslc trace_v_point_v_dir_u_mindist_v_maxdist_v_traceset.osl  
oslc trace_v_point_v_dir_u_mindist_u_maxdist_v_traceset.osl
oslc trace_v_point_v_dir_v_mindist_u_maxdist_v_traceset.osl  
oslc trace_v_point_v_dir_v_mindist_v_maxdist_v_traceset.osl

###########################################
#Varying position; Varying direction
#Uniform mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_vv_uu_out.tif -o mcout trace_vv_uu_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_vv_uu_ref.tif -o mcout trace_vv_uu_m_ref.tif

idiff trace_vv_uu_ref.tif trace_vv_uu_out.tif
idiff trace_vv_uu_m_ref.tif trace_vv_uu_m_out.tif

###########################################
#Uniform position; Uniform direction
#Varying mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_v_mindist_u_maxdist_v_traceset -od uint8 -o cout trace_vv_vu_out.tif -o mcout trace_vv_vu_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_v_mindist_u_maxdist_v_traceset -od uint8 -o cout trace_vv_vu_ref.tif -o mcout trace_vv_vu_m_ref.tif

idiff trace_vv_vu_ref.tif trace_vv_vu_out.tif
idiff trace_vv_vu_m_ref.tif trace_vv_vu_m_out.tif


###########################################
#Uniform position; Uniform direction
#Uniform mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_u_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_vv_uv_out.tif -o mcout trace_vv_uv_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_u_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_vv_uv_ref.tif -o mcout trace_vv_uv_m_ref.tif

idiff trace_vv_uv_ref.tif trace_vv_uv_out.tif
idiff trace_vv_uv_m_ref.tif trace_vv_uv_m_out.tif



###########################################
#Uniform position; Uniform direction
#Varying mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_vv_vv_out.tif -o mcout trace_vv_vv_m_out.tif
testshade -g 200 200 trace_v_point_v_dir_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_vv_vv_ref.tif -o mcout trace_vv_vv_m_ref.tif

idiff trace_vv_vv_ref.tif trace_vv_vv_out.tif
idiff trace_vv_vv_m_ref.tif trace_vv_vv_m_out.tif



