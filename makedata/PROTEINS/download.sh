wget "https://drive.google.com/uc?export=download&id=1GH0EnM_yWyUT2_zvtyfbcNbKuYYA9xBB" -O PROTEINS.zip
unzip PROTEINS.zip
mkdir -p inputs/
mv PROTEINS inputs/
rm PROTEINS.zip