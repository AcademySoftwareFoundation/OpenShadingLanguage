#!/bin/csh


cd uniform_traceset

#!/bin/csh



oslc trace_u_point_u_dir_u_mindist_v_maxdist_u_traceset.osl  
oslc trace_u_point_u_vector_u_mindist_u_maxdist_u_traceset.osl
oslc trace_u_point_u_dir_v_mindist_u_maxdist_u_traceset.osl  
oslc trace_u_point_u_vector_v_mindist_v_maxdist_u_traceset.osl

###########################################
#Uniform position; Uniform direction; Uniform traceset
#Uniform mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_uu_uu_out.tif -o mcout trace_uu_uu_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_uu_uu_ref.tif -o mcout trace_uu_uu_m_ref.tif

idiff trace_uu_uu_ref.tif trace_uu_uu_out.tif
idiff trace_uu_uu_m_ref.tif trace_uu_uu_m_out.tif

###########################################
#Uniform position; Uniform direction; Uniform traceset
#Varying mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_u_maxdist_u_traceset -od uint8 -o cout trace_uu_vu_out.tif -o mcout trace_uu_vu_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_u_maxdist_u_traceset -od uint8 -o cout trace_uu_vu_ref.tif -o mcout trace_uu_vu_m_ref.tif

idiff trace_uu_vu_ref.tif trace_uu_vu_out.tif
idiff trace_uu_vu_m_ref.tif trace_uu_vu_m_out.tif


###########################################
#Uniform position; Uniform direction; Uniform traceset
#Uniform mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_u_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_uu_uv_out.tif -o mcout trace_uu_uv_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_u_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_uu_uv_ref.tif -o mcout trace_uu_uv_m_ref.tif

idiff trace_uu_uv_ref.tif trace_uu_uv_out.tif
idiff trace_uu_uv_m_ref.tif trace_uu_uv_m_out.tif



###########################################
#Uniform position; Uniform direction; Uniform traceset
#Varying mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_uu_vv_out.tif -o mcout trace_uu_vv_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_u_traceset -od uint8 -o cout trace_uu_vv_ref.tif -o mcout trace_uu_vv_m_ref.tif

idiff trace_uu_vv_ref.tif trace_uu_vv_out.tif
idiff trace_uu_vv_m_ref.tif trace_uu_vv_m_out.tif


cd ../
cd varying_traceset


oslc trace_u_point_u_dir_u_mindist_v_maxdist_v_traceset.osl  
oslc trace_u_point_u_vector_u_mindist_u_maxdist_v_traceset.osl
oslc trace_u_point_u_dir_v_mindist_u_maxdist_v_traceset.osl  
oslc trace_u_point_u_vector_v_mindist_v_maxdist_v_traceset.osl

###########################################
#Uniform position; Uniform direction; Varying traceset
#Uniform mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uu_uu_out.tif -o mcout trace_uu_uu_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uu_uu_ref.tif -o mcout trace_uu_uu_m_ref.tif

idiff trace_uu_uu_ref.tif trace_uu_uu_out.tif
idiff trace_uu_uu_m_ref.tif trace_uu_uu_m_out.tif

###########################################
#Uniform position; Uniform direction; Varying traceset
#Varying mindist
#Uniform maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_u_maxdist_v_traceset -od uint8 -o cout trace_uu_vu_out.tif -o mcout trace_uu_vu_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_u_maxdist_v_traceset -od uint8 -o cout trace_uu_vu_ref.tif -o mcout trace_uu_vu_m_ref.tif

idiff trace_uu_vu_ref.tif trace_uu_vu_out.tif
idiff trace_uu_vu_m_ref.tif trace_uu_vu_m_out.tif


###########################################
#Uniform position; Uniform direction; Varying traceset
#Uniform mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_u_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uu_uv_out.tif -o mcout trace_uu_uv_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_u_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uu_uv_ref.tif -o mcout trace_uu_uv_m_ref.tif

idiff trace_uu_uv_ref.tif trace_uu_uv_out.tif
idiff trace_uu_uv_m_ref.tif trace_uu_uv_m_out.tif


###########################################
#Uniform position; Uniform direction; Varying traceset
#Varying mindist
#Varying maxdist
###########################################


testshade --batched -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uu_vv_out.tif -o mcout trace_uu_vv_m_out.tif
testshade -g 200 200 trace_u_point_u_vector_v_mindist_v_maxdist_v_traceset -od uint8 -o cout trace_uu_vv_ref.tif -o mcout trace_uu_vv_m_ref.tif

idiff trace_uu_vv_ref.tif trace_uu_vv_out.tif
idiff trace_uu_vv_m_ref.tif trace_uu_vv_m_out.tif


