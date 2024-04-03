MODEL='KV-PLM*'
# MODEL='KV-PLM'
IFT='0'

SCI='Sci-BERT'
KV='KV-PLM'
KV1='KV-PLM*'

cmd='--init_checkpoint'
if [ $MODEL = $KV1 ]
then
	cmd=$cmd' save_model/ckpt_KV_1.pt'
elif [$MODEL = $KV ]
then
	cmd=$cmd' save_model/ckpt_KV.pt'
fi

cd ..

mkdir finetune_save

# train and eval (We can obtain S-T Acc and T-S Acc results on test set)
python run_retriev.py $cmd --iftest $IFT

# eval (save the intermediate results for Ret/calcu_test.py, to obtain the Rec@20 results)
python run_retriev.py $cmd --iftest 1
python run_retriev.py $cmd --iftest 2

