#!/bin/csh

oslc getmessage_u_source_u_name.osl
oslc getmessage_u_source_v_name.osl
oslc getmessage_v_source_u_name.osl
oslc getmessage_v_source_v_name.osl

oslc getmessage_no_source_u_name.osl
oslc getmessage_no_source_v_name.osl

######################################
#Uniform source; Uniform name
######################################

testshade --batched -g 200 200 getmessage_u_source_u_name -od uint8 -o cout usource_uname_out.tif -o mcout m_usource_uname_out.tif 
testshade -g 200 200 getmessage_u_source_u_name -od uint8 -o cout usource_uname_ref.tif -o mcout m_usource_uname_ref.tif  

idiff usource_uname_ref.tif usource_uname_out.tif
idiff m_usource_uname_ref.tif m_usource_uname_out.tif 


######################################
#Uniform source; Varying name
######################################


testshade --batched -g 200 200 getmessage_u_source_v_name -od uint8 -o cout usource_vname_out.tif -o mcout m_usource_vname_out.tif 
testshade -g 200 200 getmessage_u_source_v_name -od uint8 -o cout usource_vname_ref.tif -o mcout m_usource_vname_ref.tif  

idiff usource_vname_ref.tif usource_vname_out.tif
idiff m_usource_vname_ref.tif m_usource_vname_out.tif 


######################################
#Varying source; Uniform name 
######################################


testshade --batched -g 200 200 getmessage_v_source_u_name -od uint8 -o cout vsource_uname_out.tif -o mcout m_vsource_uname_out.tif 
testshade -g 200 200 getmessage_v_source_u_name -od uint8 -o cout vsource_uname_ref.tif -o mcout m_vsource_uname_ref.tif  

idiff vsource_uname_ref.tif vsource_uname_out.tif
idiff m_vsource_uname_ref.tif m_vsource_uname_out.tif 


######################################
#Varying source; Varying name 
######################################


testshade --batched -g 200 200 getmessage_v_source_v_name -od uint8 -o cout vsource_vname_out.tif -o mcout m_vsource_vname_out.tif 
testshade -g 200 200 getmessage_v_source_v_name -od uint8 -o cout vsource_vname_ref.tif -o mcout m_vsource_vname_ref.tif  

idiff vsource_vname_ref.tif vsource_vname_out.tif
idiff m_vsource_vname_ref.tif m_vsource_vname_out.tif 

######################################
#No source; Uniform name (Needs further implementation in opmessage.cpp post line 200)
######################################


#testshade --batched -g 200 200 getmessage_no_source_u_name -od uint8 -o cout nosource_uname_out.tif -o mcout m_nosource_uname_out.tif 
#testshade -g 200 200 getmessage_no_source_u_name -od uint8 -o cout nosource_uname_ref.tif -o mcout m_nosource_uname_ref.tif  

#idiff nosource_uname_ref.tif nosource_uname_out.tif
#idiff m_nosource_uname_ref.tif  m_nosource_uname_out.tif

######################################
#No source; Varying name (Needs further implementation in opmessage.cpp post line 200)
######################################


#testshade --batched -g 200 200 getmessage_no_source_v_name -od uint8 -o cout nosource_vname_out.tif -o mcout m_nosource_vname_out.tif 
#testshade -g 200 200 getmessage_no_source_v_name -od uint8 -o cout nosource_vname_ref.tif -o mcout m_nosource_vname_ref.tif  

#idiff nosource_vname_ref.tif nosource_vname_out.tif
#idiff m_nosource_vname_ref.tif  m_nosource_vname_out.tif