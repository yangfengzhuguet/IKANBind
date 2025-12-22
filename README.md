# IKANBind
Protein–Nucleic Acid Binding Site Prediction Using Kolmogorov–Arnold Networks with Hypergraph Representation Learning
![image](https://github.com/yangfengzhuguet/IKANBind/blob/main/workflow.jpg)
# System requirement
python 3.8.16  
numpy 1.24.2  
pyg-lib 0.1.0+pt113cu116  
pyparsing 3.0.9  
scikit-learn 1.2.2  
six 1.16.0  
torch 1.13.1+cu116  
torch-cluster 1.6.1+pt113cu116  
torch-geometric 2.2.0   
torch-scatter 2.1.1+pt113cu116  
torch-sparse 0.6.17+pt113cu116  
torch-spline-conv 1.2.2+pt113cu116  
torchaudio 0.13.1+cu116  
torchvision 0.14.1+cu116  
urllib3  1.26.15  
wheel 0.38.4  
# ProtTrans
You need to prepare the pretrained language model ProtTrans to run GLMSite:  
Download the pretrained ProtT5-XL-UniRef50 model ([guide](https://github.com/agemagician/ProtTrans)).  
# ESMFold
The protein structures should be predicted by ESMFold to run GLMSite:  
Download the ESMFold model ([guide](https://github.com/facebookresearch/esm))  
# Run GLMSite for prediction
Simply run:  
```
python predict.py --dataset_path ../Example/structure_data/ --feature_path ../Example/prottrans/ --input_path ../Example/demo.pkl
```
And the prediction results will be saved in  
```
../Example/results
```


