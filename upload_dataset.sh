cd  /ssd_scratch/cvit/avijit/data/
wget https://s3-us-west-2.amazonaws.com/ai2-vision-datasets/ForScene_dataset/ForScene.tar.gz
wget http://www.doc.ic.ac.uk/%7Eahanda/SUNRGBD-train_images.tgz
wget http://www.doc.ic.ac.uk/%7Eahanda/SUNRGBD-test_images.tgz
tar xvf ForScene.tar.gz
mkdir -p SUNRGBD_train
mkdir -p SUNRGBD_test
tar xvf SUNRGBD-train_images.tgz -C SUNRGBD_train
tar xvf SUNRGBD-test_images.tgz -C SUNRGBD_test
cd SUNRGBD_train 
find . -name '*.jpg' | awk 'BEGIN{ a=5051 }{ printf "mv \"%s\" %05d.png\n", $0, a++ }'| bash
cd ../SUNRGBD_test
find . -name '*.jpg' | awk 'BEGIN{ a=1}{ printf "mv \"%s\" %05d.png\n", $0, a++ }'| bash
cd ..
mkdir -p rgbimages/
mv  SUNRGBD_train/* rgbimages/
mv SUNRGBD_test/* rgbimages/
mv rgbimages data_release
rm -rf SUNRGBD_*
rm -rf *.tgz
rm -rf ForScene.tar.gz
cd /home/avijit.d/forscene/
