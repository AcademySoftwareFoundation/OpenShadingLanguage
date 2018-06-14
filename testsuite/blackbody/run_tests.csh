!#/bin/csh


oslc blackbody_u_temperature.osl  
oslc blackbody_v_temperature.osl

#############################################
#Varying temperature
#############################################
testshade --batched -g 200 200 blackbody_v_temperature -od uint8 -o Cout v_temp_out.tif -o mCout m_v_temp_out.tif 
testshade -g 200 200 blackbody_v_temperature -od uint8 -o Cout v_temp_ref.tif -o mCout m_v_temp_ref.tif 
idiff v_temp_ref.tif v_temp_out.tif 
idiff m_v_temp_ref.tif  m_v_temp_out.tif 

#############################################
#Uniform temperature 
#############################################
testshade --batched -g 200 200 blackbody_u_temperature -od uint8 -o Cout u_temp_out.tif -o mCout m_u_temp_out.tif 
testshade -g 200 200 blackbody_u_temperature -od uint8 -o Cout u_temp_ref.tif -o mCout m_u_temp_ref.tif 
idiff u_temp_ref.tif u_temp_out.tif 
idiff m_u_temp_ref.tif  m_u_temp_out.tif 
