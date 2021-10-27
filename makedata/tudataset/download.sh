wget "https://drive.google.com/uc?export=download&id=11un3h1Rnm8Q7XUSQTHM4MGeVgkKy4Yl4" -O tudataset.zip
unzip tudataset.zip
mkdir -p inputs/
mv tudataset/* inputs/
rm tudataset.zip