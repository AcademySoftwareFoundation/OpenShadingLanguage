#!/bin/csh

cd uniform_pos_uniform_dir

oslc trace_u_point_u_dir_u_mindist_v_maxdist.osl  
oslc trace_u_point_u_vector_u_mindist_u_maxdist.osl
oslc trace_u_point_u_dir_v_mindist_u_maxdist.osl  
oslc trace_u_point_u_vector_v_mindist_v_maxdist.osl

###########################################
#Uniform position; Uniform direction
#Uniform mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist -od uint8 -o cout trace_uu_uu_out.tif -o mcout trace_uu_uu_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist -od uint8 -o cout trace_uu_uu_ref.tif -o mcout trace_uu_uu_m_ref.tif

idiff trace_uu_uu_ref.tif trace_uu_uu_out.tif
idiff trace_uu_uu_m_ref.tif trace_uu_uu_m_out.tif

###########################################
#Uniform position; Uniform direction
#Varying mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_u_maxdist -od uint8 -o cout trace_uu_vu_out.tif -o mcout trace_uu_vu_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_u_maxdist -od uint8 -o cout trace_uu_vu_ref.tif -o mcout trace_uu_vu_m_ref.tif

idiff trace_uu_vu_ref.tif trace_uu_vu_out.tif
idiff trace_uu_vu_m_ref.tif trace_uu_vu_m_out.tif


###########################################
#Uniform position; Uniform direction
#Uniform mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_u_mindist_v_maxdist -od uint8 -o cout trace_uu_uv_out.tif -o mcout trace_uu_uv_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_u_mindist_v_maxdist -od uint8 -o cout trace_uu_uv_ref.tif -o mcout trace_uu_uv_m_ref.tif

idiff trace_uu_uv_ref.tif trace_uu_uv_out.tif
idiff trace_uu_uv_m_ref.tif trace_uu_uv_m_out.tif



###########################################
#Uniform position; Uniform direction
#Varying mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist -od uint8 -o cout trace_uu_vv_out.tif -o mcout trace_uu_vv_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist -od uint8 -o cout trace_uu_vv_ref.tif -o mcout trace_uu_vv_m_ref.tif

idiff trace_uu_vv_ref.tif trace_uu_vv_out.tif
idiff trace_uu_vv_m_ref.tif trace_uu_vv_m_out.tif

