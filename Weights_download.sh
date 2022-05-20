#!/bin/bash
#!/bin/bash

ALEX=alexnet.weights
VGG=vgg16.weights
TDARK=tinydark.weights
MONET=mobilenet_v1_72.weights
TYOLO=yolov2-tiny-voc.weights

#alexnet
#11Sa_4mUj8DFbbYuJDAFVfbTln90vARMv
if [ ! -f "$ALEX" ]; then
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11Sa_4mUj8DFbbYuJDAFVfbTln90vARMv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11Sa_4mUj8DFbbYuJDAFVfbTln90vARMv" -O $ALEX && rm -rf ~/cookies.txt
fi

#vgg16
#1yJaV52csVb3Gwe_yietXobNb3VrA3EZZ
if [ ! -f "$VGG" ]; then
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yJaV52csVb3Gwe_yietXobNb3VrA3EZZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yJaV52csVb3Gwe_yietXobNb3VrA3EZZ" -O $VGG && rm -rf ~/cookies.txt
fi


#tinydark
#1PrRr-cmM8ZYKjF9hTN3aiW4VYFi35Ove
if [ ! -f "$TDARK" ]; then
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PrRr-cmM8ZYKjF9hTN3aiW4VYFi35Ove' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PrRr-cmM8ZYKjF9hTN3aiW4VYFi35Ove" -O $TDARK && rm -rf ~/cookies.txt
fi

#yolov2tiny
#1ZY7WG98huYyVd6LTq_UyYnVbhI-r8qeY
if [ ! -f "$TYOLO" ]; then
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZY7WG98huYyVd6LTq_UyYnVbhI-r8qeY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZY7WG98huYyVd6LTq_UyYnVbhI-r8qeY" -O $TYOLO && rm -rf ~/cookies.txt
fi

#mobilenet
#1XmrbHrnDIWkBkrk4MiyWz8IbsENU7hoT
if [ ! -f "$MONET" ]; then
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XmrbHrnDIWkBkrk4MiyWz8IbsENU7hoT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XmrbHrnDIWkBkrk4MiyWz8IbsENU7hoT" -O $MONET && rm -rf ~/cookies.txt
fi
