oslc pow_u_vector_base_v_df_exp.osl
oslc pow_v_dv_base_u_float_exp.osl
oslc pow_v_dv_base_v_df_exp.osl
oslc pow_u_vector_base_v_float_exp.osl
oslc pow_u_vector_base_u_float_exp.osl
//pow_u_vector_base_v_df_exp.osl


echo "\n"
echo "*******************"
echo "Uniform vector base, varying float exponent"
echo "*******************"

testshade --batched  -g 200 200 pow_u_vector_base_v_float_exp  -od uint8 -o res u_vec_base_v_float_out.tif -o res_m m_u_vec_base_v_float_out.tif
testshade -g 200 200 pow_u_vector_base_v_float_exp -od uint8 -o res u_vec_base_v_float_ref.tif -o res_m m_u_vec_base_v_float_ref.tif

idiff u_vec_base_v_float_ref.tif u_vec_base_v_float_out.tif
idiff m_u_vec_base_v_float_out.tif m_u_vec_base_v_float_ref.tif


echo "\n"
echo "************************************************"
echo "Varying dual vector base, varying dual float exp"
echo "************************************************"

testshade --batched  -g 200 200 pow_v_dv_base_v_df_exp -od uint8 -o res vdv_vec_base_vdf_exp_out.tif \
                                                                                      -o Dxres vdv_vec_base_vdf_exp_dx_out.tif \
                                                                                      -o res_m m_vdv_vec_base_vdf_exp_out.tif\
                                                                                      -o Dxres_m m_vdv_vec_base_vdf_exp_dx_out.tif
                                                                                      

                                                                                      
testshade -g 200 200 pow_v_dv_base_v_df_exp -od uint8 -o res vdv_vec_base_vdf_exp_ref.tif \
                                                    -o Dxres vdv_vec_base_vdf_exp_dx_ref.tif \
                                                    -o res_m m_vdv_vec_base_vdf_exp_ref.tif\
                                                    -o Dxres_m m_vdv_vec_base_vdf_exp_dx_ref.tif
                                                    
idiff vdv_vec_base_vdf_exp_ref.tif vdv_vec_base_vdf_exp_out.tif
idiff vdv_vec_base_vdf_exp_dx_ref.tif vdv_vec_base_vdf_exp_dx_out.tif
idiff m_vdv_vec_base_vdf_exp_ref.tif m_vdv_vec_base_vdf_exp_out.tif
idiff m_vdv_vec_base_vdf_exp_dx_ref.tif m_vdv_vec_base_vdf_exp_dx_out.tif


echo "\n"
echo "************************************************"
echo "Varying dual vector base, uniform float exp"
echo "************************************************"

testshade --batched  -g 200 200 pow_v_dv_base_u_float_exp -od uint8 -o res vdv_vec_base_u_f_exp_out.tif \
                                                                                      -o Dxres vdv_vec_base_u_f_exp_dx_out.tif \
                                                                                      -o res_m m_vdv_vec_base_u_f_exp_out.tif\
                                                                                      -o Dxres_m m_vdv_vec_base_u_f_exp_dx_out.tif
                                                                                      

                                                                                      
testshade -g 200 200 pow_v_dv_base_u_float_exp -od uint8 -o res vdv_vec_base_u_f_exp_ref.tif \
                                                                     -o Dxres vdv_vec_base_u_f_exp_dx_ref.tif \
                                                                     -o res_m m_vdv_vec_base_u_f_exp_ref.tif\
                                                                     -o Dxres_m m_vdv_vec_base_u_f_exp_dx_ref.tif
                                                                                                                                                                            

                                                    
idiff vdv_vec_base_u_f_exp_ref.tif vdv_vec_base_u_f_exp_out.tif
idiff vdv_vec_base_u_f_exp_dx_ref.tif vdv_vec_base_u_f_exp_dx_out.tif
idiff m_vdv_vec_base_u_f_exp_ref.tif m_vdv_vec_base_u_f_exp_out.tif
idiff m_vdv_vec_base_u_f_exp_dx_ref.tif m_vdv_vec_base_u_f_exp_dx_out.tif

echo "\n"
echo "************************************************"
echo "Uniform vector base, varying  dual float exp"
echo "************************************************"



testshade --batched  -g 200 200 pow_u_vec_base_v_df_exp -od uint8 -o res u_vec_base_v_df_exp_out.tif \
                                                                  -o Dxres u_vec_base_v_df_exp_dx_out.tif \
                                                                  -o res_m m_u_vec_base_v_df_exp_out.tif\
                                                                  -o Dxres_m m_u_vec_base_v_df_exp_dx_out.tif
                                                                  
                                                                  
                                                                  
testshade  -g 200 200 pow_u_vec_base_v_df_exp -od uint8 -o res u_vec_base_v_df_exp_ref.tif \
                                                                  -o Dxres u_vec_base_v_df_exp_dx_ref.tif \
                                                                  -o res_m m_u_vec_base_v_df_exp_ref.tif\
                                                                  -o Dxres_m m_u_vec_base_v_df_exp_dx_ref.tif
                                                                  

                                                                  
idiff u_vec_base_v_df_exp_ref.tif u_vec_base_v_df_exp_out.tif
idiff u_vec_base_v_df_exp_dx_ref.tif  u_vec_base_v_df_exp_dx_out.tif
idiff m_u_vec_base_v_df_exp_ref.tif m_u_vec_base_v_df_exp_out.tif
idiff m_u_vec_base_v_df_exp_dx_ref.tif m_u_vec_base_v_df_exp_dx_out.tif

echo "\n"
echo "************************************************"
echo "Uniform vector base, uniform float exp"
echo "************************************************"

 
testshade --batched  -g 200 200 pow_u_vector_base_u_float_exp  -od uint8 -o res u_vec_base_u_float_out.tif -o res_m m_u_vec_base_u_float_out.tif
testshade -g 200 200 pow_u_vector_base_u_float_exp -od uint8 -o res u_vec_base_u_float_ref.tif -o res_m m_u_vec_base_u_float_ref.tif

idiff u_vec_base_u_float_ref.tif u_vec_base_u_float_out.tif
idiff m_u_vec_base_u_float_out.tif m_u_vec_base_u_float_ref.tif

