# 
pytorch 1.9.0 + cu111
python 3.6.6
numpy 1.19.5

dota_devkit
swig and setup follow readme

Polygon for text eval

prepare image and modelfile then run eval
[HRSCmodel-kb96](https://pan.baidu.com/s/1SPT3_hZfKPb1nBniUJzoIQ  
)  
res101
1280*720 88.9
740*500 90.6

prepare image and modelfile then run eval
[M500model-c280](https://pan.baidu.com/s/1cPjHDCA80xD91Ya9CBcCSg  
)
res50
1280*720 without instance segmentation 65.6 F
res 101 with instance segmentation +11%F

prepare image and modelfile then run eval
[dotamodel-d25g]
res101
cut 600 resize 800
800*800 total map 73.5
we will sharing a res50 model with multi-scale
