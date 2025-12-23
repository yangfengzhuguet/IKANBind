# IKANBind：Protein–Nucleic Acid Binding Site Prediction Using Kolmogorov–Arnold Networks with Hypergraph Representation Learning
![image](https://github.com/yangfengzhuguet/IKANBind/blob/main/workflow.jpg)
# ProtTrans
You need to prepare the pretrained language model ProtTrans to run IKANBind:  
Download the pretrained ProtT5-XL-UniRef50 model ([guide](https://github.com/agemagician/ProtTrans)).  
# ESM-2
You need to prepare the pretrained language model ESM-2 to run IKANBind:  
Download the pretrained ESM-2 model ([guide](https://github.com/facebookresearch/esm)).  
# ESMFold
The protein structures should be predicted by ESMFold to run IKANBind:  
Download the ESMFold model ([guide](https://github.com/facebookresearch/esm))  
# Run IKANBind for prediction
Simply run:  
```
Please download the corresponding model parameters from the link (https://drive.google.com/drive/folders/1fE41iSYFBWfxkYEgjw3kzm1AAnzWJ3oyo)
Then run：
python main.py
please note：The above program loads the DNA/RNA model parameters for testing by default. If you want to retrain the model, please set the flag in main.py to train.
```
# contact
Yongxian Fan (yongxian.fan@gmail.com)  
Xiaoyong Pan (2008xypan@sjtu.edu.cn)


