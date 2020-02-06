for a in {1..9}
do 
python bls2017.py --verbose compress ../../dataset/kodak24/kodim0${a}.png --checkpoint 17-clic
done 

for a in {10..24}
do 
python bls2017.py --verbose compress ../../dataset/kodak24/kodim${a}.png --checkpoint 17-clic
done 