
oslc a_u_b_u.osl
oslc a_v_b_u.osl
oslc a_u_b_v.osl
oslc a_v_b_v.osl

testshade --batched -g 200 200 -od uint8 a_v_b_u -o cout vu_batched.tif -o mcout vu_batched_m.tif
testshade -g 200 200 -od uint8 a_v_b_u -o cout vu_ref.tif -o mcout vu_ref_m.tif
idiff vu_ref.tif vu_batched.tif
idiff vu_ref_m.tif vu_batched_m.tif

testshade --batched -g 200 200 -od uint8 a_u_b_v -o cout uv_batched.tif -o mcout uv_batched_m.tif
testshade -g 200 200 -od uint8 a_u_b_v -o cout uv_ref.tif -o mcout uv_ref_m.tif
idiff uv_ref.tif uv_batched.tif
idiff uv_ref_m.tif uv_batched_m.tif



testshade --batched -g 200 200 -od uint8 a_v_b_v -o cout vv_batched.tif -o mcout vv_batched_m.tif
testshade -g 200 200 -od uint8 a_v_b_v -o cout vv_ref.tif -o mcout vv_ref_m.tif
idiff vv_ref.tif vv_batched.tif
idiff vv_ref_m.tif vv_batched_m.tif



testshade --batched -g 200 200 -od uint8 a_u_b_u -o cout uu_batched.tif -o mcout uu_batched_m.tif
testshade -g 200 200 -od uint8 a_u_b_u -o cout uu_ref.tif -o mcout uu_ref_m.tif
idiff uu_ref.tif uu_batched.tif
idiff uu_ref_m.tif uu_batched_m.tif


