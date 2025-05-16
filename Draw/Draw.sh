#HLN:                 s=100
#HT:                  s=80
#MISAR(S1 or S2):     s=140
#Mouse_Brain:         s=40
#CITE(Mouse_Spleen):  s=100
#CITE(Mouse_Spleen)_s2:  s=80
#SPOTS(Mouse_Thymus): s=50


python Draw_umap_spatial.py --adata_path [Adata_path] --key [feat_key] --GT_Path [GT_Path if exist] --s [s] --Method [Method Name] --Dataset [Dataset Name]
