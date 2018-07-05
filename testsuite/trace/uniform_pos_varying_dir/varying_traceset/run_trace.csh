#!/bin/csh



oslc trace_u_point_v_dir_u_mindist_v_maxdist_v_traceset.osl  
oslc trace_u_point_v_vector_u_mindist_u_maxdist_v_traceset.osl
oslc trace_u_point_v_dir_v_mindist_u_maxdist_v_traceset.osl  
oslc trace_u_point_v_vector_v_mindist_v_maxdist_v_traceset.osl


###########################################
#Uniform position; Varying direction; Varying traceset
#Uniform mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_v_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uv_uu_out.tif -o mcout trace_uv_uu_m_out.tif
testshade -g 200 200 trace_u_point_v_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uv_uu_ref.tif -o mcout trace_uv_uu_m_ref.tif

idiff trace_uv_uu_ref.tif trace_uv_uu_out.tif
idiff trace_uv_uu_m_ref.tif trace_uv_uu_m_out.tif

###########################################
#Uniform position; Varying direction
#Varying mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_v_vector_v_mindist_u_maxdist_v_traceset -od uint8 -o cout trace_uv_vu_out.tif -o mcout trace_uv_vu_m_out.tif
testshade -g 200 200 trace_u_point_v_vector_v_mindist_u_maxdist_v_traceset -od uint8 -o cout trace_uv_vu_ref.tif -o mcout trace_uv_vu_m_ref.tif

idiff trace_uv_vu_ref.tif trace_uv_vu_out.tif
idiff trace_uv_vu_m_ref.tif trace_uv_vu_m_out.tif


###########################################
#Uniform position; Varying direction
#Uniform mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_v_vector_u_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uv_uv_out.tif -o mcout trace_uv_uv_m_out.tif
testshade -g 200 200 trace_u_point_v_vector_u_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uv_uv_ref.tif -o mcout trace_uv_uv_m_ref.tif

idiff trace_uv_uv_ref.tif trace_uv_uv_out.tif
idiff trace_uv_uv_m_ref.tif trace_uv_uv_m_out.tif



###########################################
#Uniform position; Varying direction
#Varying mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_v_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uv_vv_out.tif -o mcout trace_uv_vv_m_out.tif
testshade -g 200 200 trace_u_point_v_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uv_vv_ref.tif -o mcout trace_uv_vv_m_ref.tif

idiff trace_uv_vv_ref.tif trace_uv_vv_out.tif
idiff trace_uv_vv_m_ref.tif trace_uv_vv_m_out.tif

