# StegaStamp using Pytorch
#### Contributers: Dan Epshtien and Neta Becker. Based on Jisong Xie's work

### Goals: 
* Develop a tool that encrypts information into a natural image based on existing code from Berkeley university. 
* Improve the algorithm performance in order to receive optimal output images along with good decoding performance. 

### Notations:
* Original image - the image before encoding
* Encoded image - The image after encoding
* Residual image - the image that is received by substracting the original image from the encoded image (Encoded image - Original image). meaning the values that were added to the original image during the encoding stage.

## Results
### Using different loss functions, we managed to receive those results

#### Left to right: residual image, encoded image, original image:
![image](https://github.com/netabecker/Stegastamp_pytorch_version/assets/83274903/36e819c5-1109-4d81-93ef-205ad1da96d2)

A few more examples (Left to right: residual image, encoded image, original image)

<img width="591" alt="image" src="https://github.com/netabecker/Stegastamp_pytorch_version/assets/83274903/4d2518f3-5327-4d80-9d18-da5f11b1ab63">
<img width="605" alt="image" src="https://github.com/netabecker/Stegastamp_pytorch_version/assets/83274903/902eb630-93a4-4943-86ba-6b99b4d91480">

### More notations:
* secret loss - The loss function of the encoding
* decipher indicator - Graph that depicts the number of images the decoder managed to decipher out of each batch of 4 images

As seen in the graphs below, there is a trade-off between the two - if secret loss value is low than the decipher indicator is low (meaning we are able to decipher less images out of each batch) and vice versa.
![image](https://github.com/netabecker/Stegastamp_pytorch_version/assets/83274903/c4756dfa-3ada-43fd-8961-16a9fdd4b91c)

## Trained model
Per request - I added the trained models of the encoder and decoder. You can use them by running the following commands:

<ins>To encode:</ins>
python encode_image.py ||path to the trained encoder|| --images_dir=||path to images dir|| --save_dir=||path to location of wanted save dir||

<ins>To decode:</ins>
python decode_image.py ||path to the trained decoder|| --images_dir=||path to the encoded images dir||

