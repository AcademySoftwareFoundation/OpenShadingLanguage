#!/bin/csh

oslc str_concat.osl
oslc str_endswith.osl
oslc str_getchar.osl
oslc str_hash.osl
oslc str_startswith.osl
oslc str_strlen.osl
oslc str_substr.osl
oslc str_stof.osl
oslc str_stoi.osl

################################
##osl_concat
################################
testshade --batched -g 200 200 str_concat -od uint8 -o res concat_out.tif -o res_m concat_m_out.tif
testshade -g 200 200 str_concat -od uint8 -o res concat_ref.tif -o res_m concat_m_ref.tif

idiff concat_ref.tif concat_out.tif
idiff concat_m_ref.tif concat_m_ref.tif


################################
##osl_stoi
################################

testshade --batched -g 200 200 str_stoi -od uint8 -o res stoi_out.tif -o res_m stoi_m_out.tif
testshade -g 200 200 str_stoi -od uint8 -o res stoi_ref.tif -o res_m stoi_m_ref.tif

idiff stoi_ref.tif stoi_out.tif
idiff stoi_m_ref.tif stoi_m_out.tif


################################
###osl_endswith
################################

testshade --batched -g 200 200 str_endswith -od uint8 -o res_t endswith_t_out.tif -o res_f endswith_f_out.tif \
                                                -o res_t_m endswith_t_m_out.tif -o res_f_m endswith_f_m_out.tif
                                                
                                                
testshade -g 200 200 str_endswith -od uint8 -o res_t endswith_t_ref.tif -o res_f endswith_f_ref.tif \
                                               -o res_t_m endswith_t_m_ref.tif -o res_f_m endswith_f_m_ref.tif



idiff endswith_t_ref.tif endswith_t_out.tif 
idiff endswith_f_ref.tif endswith_f_out.tif 
idiff endswith_t_m_ref.tif endswith_t_m_out.tif 
idiff endswith_f_m_ref.tif endswith_f_m_out.tif

################################
###osl_getchar
################################

testshade --batched -g 200 200 str_getchar -od uint8 str_getchar  -o res_t1 getchar_t1_out.tif -o res_t2 getchar_t2_out.tif \
                                                                  -o res_f1 getchar_f1_out.tif -o res_f2 getchar_f2_out.tif \
                                                                  -o res_t1_m getchar_t1_m_out.tif -o res_t2_m getchar_t2_m_out.tif \
                                                                  -o res_f1_m getchar_f1_m_out.tif -o res_f2_m getchar_f2_m_out.tif
                                                                  
testshade  -g 200 200 str_getchar -od uint8 str_getchar  -o res_t1 getchar_t1_ref.tif -o res_t2 getchar_t2_ref.tif \
                                                                  -o res_f1 getchar_f1_ref.tif -o res_f2 getchar_f2_ref.tif \
                                                                  -o res_t1_m getchar_t1_m_ref.tif -o res_t2_m getchar_t2_m_ref.tif \
                                                                  -o res_f1_m getchar_f1_m_ref.tif -o res_f2_m getchar_f2_m_ref.tif 
                                                                   
idiff getchar_t1_ref.tif getchar_t1_out.tif
idiff getchar_t2_ref.tif getchar_t2_out.tif
idiff getchar_f1_ref.tif getchar_f1_out.tif
idiff getchar_f2_ref.tif getchar_f2_out.tif

#Masked
idiff getchar_t1_m_ref.tif getchar_t1_m_out.tif
idiff getchar_t2_m_ref.tif getchar_t2_m_out.tif
idiff getchar_f1_m_ref.tif getchar_f1_m_out.tif
idiff getchar_f2_m_ref.tif getchar_f2_m_out.tif

################################
###osl_hash
################################

testshade --batched -g 200 200 str_hash -od uint8 -o res hash_out.tif -o res_m hash_m_out.tif
testshade -g 200 200 str_hash -od uint8  -o res hash_ref.tif -o res_m hash_m_ref.tif

idiff hash_ref.tif hash_out.tif
idiff hash_m_ref.tif hash_m_out.tif

################################
###osl_startswith
################################

testshade --batched -g 200 200 str_startswith -od uint8 -o res_t startswith_t_out.tif -o res_f startswith_f_out.tif \
                                              -o res_t_m startswith_t_m_out.tif -o res_f_m startswith_f_m_out.tif

testshade -g 200 200 str_startswith -od uint8 -o res_t startswith_t_ref.tif -o res_f startswith_f_ref.tif \
                                              -o res_t_m startswith_t_m_ref.tif -o res_f_m startswith_f_m_ref.tif

idiff  startswith_t_ref.tif startswith_t_out.tif
idiff  startswith_f_ref.tif startswith_f_out.tif

#Masked
idiff  startswith_t_m_ref.tif  startswith_t_m_out.tif 
idiff  startswith_f_m_ref.tif  startswith_f_m_out.tif

################################
###osl_stof
################################

testshade --batched -g 200 200 str_stof -od uint8 -o res stof_out.tif -o res_m stof_m_out.tif
testshade -g 200 200 str_stof -od uint8 -o res stof_ref.tif -o res_m stof_m_ref.tif

idiff stof_ref.tif stof_out.tif
idiff stof_m_ref.tif stof_m_out.tif


################################
###osl_strlen
################################

testshade --batched -g 200 200 str_strlen -od uint8 -o res strlen_out.tif -o res_m strlen_m_out.tif
testshade -g 200 200 str_strlen -od uint8 -o res strlen_ref.tif -o res_m strlen_m_ref.tif

idiff strlen_ref.tif strlen_out.tif
idiff strlen_m_ref.tif strlen_m_out.tif 

################################
###osl_substr
################################

testshade --batched -g 200 200 str_substr -od uint8 -o res sub_out.tif -o res1 sub1_out.tif -o res2 sub2_out.tif \
                                                    -o res_m sub_m_out.tif -o res1_m sub1_m_out.tif -o res2_m sub2_m_out.tif

testshade -g 200 200 str_substr -od uint8 -o res sub_ref.tif -o res1 sub1_ref.tif -o res2 sub2_ref.tif \
                                                    -o res_m sub_m_ref.tif -o res1_m sub1_m_ref.tif -o res2_m sub2_m_ref.tif
                                                    
idiff sub_ref.tif  sub_out.tif                                                    
idiff sub1_ref.tif sub1_out.tif    
idiff sub2_ref.tif sub2_out.tif

#Masked
idiff sub_m_ref.tif sub_m_out.tif
idiff sub1_m_ref.tif sub1_m_out.tif
idiff sub2_m_ref.tif sub2_m_out.tif








