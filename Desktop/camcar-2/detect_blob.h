#ifndef _DETECT_BLOB_H_
#define _DETECT_BLOB_H_
//======================================================================
//
// Module that provides an interface to the Raspberry PI camera and
// the blob tracking library quickblob.
//
// author: Raimund Kirner, University of Hertfordshire
//         initial version: Dec.2016
//
// license: GNU LESSER GENERAL PUBLIC LICENSE
//          Version 2.1, February 1999
//          (for details see LICENSE file)
//
// This module will be linked as a module to the main program, by inlcuding
// also the local library quickblob, which can be obtained from:
//          https://github.com/keenerd/quickblob
//
//======================================================================

#include "quickblob.h"

//======================================================================
// Data structure of still images
typedef struct JImage {
  int w; // image width (x)
  int h; // image height (y)
  int numChannels; // 3 = RGB, 4 = RGBA
  unsigned char *data;
} TJImage;

// macro to access raw data of loaded images
#define JImageDATA(pimg,x,y,c) ((pimg)->data[ (y)*(pimg)->w*(pimg)->numChannels + (x)*(pimg)->numChannels + (c) ])

// Data structure for blob search results
typedef struct BlobSearch {
  struct blob blob;  // detailed blob data (see quickblob.h)
  double halign;  // horizontal alignment of blob (-1..max left, +1..max right, 0..middle);
  double valign;  // vertical alignment of blob (-1..max bottom, +1..max top, 0..middle);
  int size; // if size==0 then no blob has been found
  TJImage *pimg;  // pointer to the image data this blob belongs to
} TBlobSearch;


//======================================================================
// cameraSearchBlob():
// Take a picture and searches there for a blob with the given color.
// If no blob is found, the size is set to sero.
// Mem: This function automatically deletes the the image data.
TBlobSearch cameraSearchBlob(const char color[3]);

// imageSearchBlob():
// Search in an image for the maximum large blob with the given color.
// If no blob is found, the size is set to sero.
TBlobSearch imageSearchBlob(const char color[3], TJImage *pimg);

// read_JPEG_image():
// Function to read jpeg image data (using libjpeg)
// Mem: The data buffer of the returned image gets overwritten on each call.
TJImage read_JPEG_image (FILE *file);

// readJpegImageFromFile():
// Function to read jpeg image data (using libjpeg)
// Mem: The data buffer of the returned image gets overwritten on each call.
TJImage readJpegImageFromFile (const char *fname);

// capturePhotoToFile():
// Take a picture via RasperiPI camera and save it as a .jpg file.
int capturePhotoToFile(const char *fname);

// capturePhoto():
// Take a picture via RasperiPI camera and return the raw image data
// Mem: The meory for the image data needs to be explicitly freed.
TJImage capturePhoto();

// Function to save a loaded image as JPEG file
// quality: integer 0..100
void writeImageAsJPEG(TJImage *pimg, const char *fname, int quality);

// Function to mark a loaded image with a blob and save it as JPEG file
void writeImageWithBlobAsJPEG(TBlobSearch blobsearch, const char *fname, int quality);

// Function to save a loaded image as a CSV (comma separated value) 
// text file.  This function might be useful to analyse the light situation
// for the camera.  Note that this may produce much large files than the
// original image.
void writeImageAsCSV(TJImage *pimg, const char *fname);


#endif /* _DETECT_BLOB_H_ */
