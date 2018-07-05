oslc tanh_u_float.osl #Uniform float
oslc tanh_v_float.osl #Varying float
oslc tanh_v_float_df.osl #Varying float, but ask for Derivs

oslc tanh_v_vector.osl #Varying vector
oslc tanh_u_vector.osl #Uniform vector
oslc tanh_v_vector_dv.osl #Varying vector, but ask for Derivs

#############################
#Varying float 
#
#osl_tanh_w16fw16f
#osl_tanh_w16fw16f_masked
#############################

testshade --batched -g 200 200 tanh_v_float -od uint8 -o res vwfwf_out.tif -o res_m m_vwfwf_out.tif
testshade -g 200 200 tanh_v_float -od uint8 -o res vwfwf_ref.tif -o res_m m_vwfwf_ref.tif

idiff vwfwf_ref.tif vwfwf_out.tif
idiff m_vwfwf_ref.tif m_vwfwf_out.tif

#############################
#Uniform float 
#
#osl_tanh_w16fw16f
#osl_tanh_w16fw16f_masked

#############################

testshade --batched -g 200 200 tanh_u_float -od uint8 -o res uwfwf_out.tif -o res_m m_uwfwf_out.tif
testshade -g 200 200 tanh_u_float -od uint8 -o res uwfwf_ref.tif -o res_m m_uwfwf_ref.tif

idiff uwfwf_ref.tif uwfwf_out.tif
idiff m_uwfwf_ref.tif m_uwfwf_out.tif

#############################
#Varying float, df
#
#osl_tanh_w16dfw16df
#osl_tanh_w16dfw16df_masked
#############################
testshade --batched -g 200 200 tanh_v_float_df -od uint8 -o res wdfwdf_out.tif -o Dxres wdfwdf_dx_out.tif -o res_m m_wdfwdf_out.tif -o Dxres_m m_wdfwdf_dx_out.tif

testshade  -g 200 200 tanh_v_float_df -od uint8 -o res wdfwdf_ref.tif -o Dxres wdfwdf_dx_ref.tif -o res_m m_wdfwdf_ref.tif -o Dxres_m m_wdfwdf_dx_ref.tif

idiff wdfwdf_out.tif wdfwdf_ref.tif
idiff wdfwdf_dx_out.tif  wdfwdf_dx_ref.tif

#Masked
idiff m_wdfwdf_out.tif m_wdfwdf_ref.tif
idiff m_wdfwdf_dx_out.tif m_wdfwdf_dx_ref.tif 

#############################
#Varying vector
#
#osl_tanh_w16vw16v
#osl_tanh_w16vw16v_masked
#############################

testshade --batched -g 200 200 tanh_v_vector -od uint8 -o res vwvwv_out.tif -o res_m m_vwvwv_out.tif
testshade -g 200 200 tanh_v_vector -od uint8 -o res vwvwv_ref.tif -o res_m m_vwvwv_ref.tif

idiff vwvwv_ref.tif vwvwv_out.tif
idiff m_vwvwv_ref.tif m_vwvwv_out.tif

#############################
#Uniform vector
#osl_tanh_vv
#osl_tanh_w16vw16v_masked

#############################

testshade --batched -g 200 200 tanh_u_vector -od uint8 -o res uvwvwv_out.tif -o res_m m_uvwvwv_out.tif
testshade -g 200 200 tanh_u_vector -od uint8 -o res uvwvwv_ref.tif -o res_m m_uvwvwv_ref.tif

idiff uvwvwv_ref.tif uvwvwv_out.tif
idiff m_uvwvwv_ref.tif m_uvwvwv_out.tif


#############################
#Varying vector, dv
#

#############################

testshade --batched -g 200 200 tanh_v_vector_dv -od uint8 -o res wdvwdv_out.tif -o Dxres wdvwdv_dx_out.tif -o res_m m_wdvwdv_out.tif -o Dxres_m m_wdvwdv_dx_out.tif
testshade  -g 200 200 tanh_v_vector_dv -od uint8 -o res wdvwdv_ref.tif -o Dxres wdvwdv_dx_ref.tif -o res_m m_wdvwdv_ref.tif -o Dxres_m m_wdvwdv_dx_ref.tif

idiff wdvwdv_out.tif wdvwdv_ref.tif
idiff wdvwdv_dx_out.tif  wdvwdv_dx_ref.tif

#Masked
idiff m_wdvwdv_out.tif m_wdvwdv_ref.tif
idiff m_wdvwdv_dx_out.tif m_wdvwdv_dx_ref.tif

