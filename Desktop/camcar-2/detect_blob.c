
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


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <jpeglib.h>
#include <jerror.h>
#include <assert.h>
#include <string.h>
#include "detect_blob.h"
#include "quickblob.h"

#define max(a,b)  ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b)  ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

typedef struct QuickBlob {
  TJImage *pimg;
  char ref[3]; // color RGB reference value for blob filtering;
  int frame; // single-imgage app, so not used;
  double ref_rel[3]; // reference value, normalised to red component;
  struct blob blob_max; // maximal blob to be found;
} TQuickBlob;

#define BLOB_MATCH(ref,dat) (((ref)*0.9 <= (dat)) && ((dat) <= min(255,(ref)*1.1)))

// raspistill --help
// -w 200 -h 200  # set image size (width, height)
// -ss <val> # set shutter speed to <val> us
// -t 1 # time in ms before image is taken
// -awb fluorescent # set white balance
// --mode, -md 7 # size: 640x480, frame rate: 60.1-90fps (max pixels: -md 2)
// -n, -nopreview # don't show preview window
// -rot 90 # rotate image (if camera mounted with cable leaving right)
// -rot 180 # rotate image (if camera mounted with cable leaving downwards)
// -o <file.jpg>  # write to <file.jpg>
// -o -  # write to stdout
#define CAMERA_CMD "raspistill -w 200 -h 200 -t 1 -awb fluorescent --nopreview --mode 7 -rot 270" // use if needed: -rot 0/90/180/270


//======================================================================
// some local hookback functions to be defined for quickblob:
//======================================================================
void log_blob_hook(void* user_struct, struct blob* b);
int init_pixel_stream_hook(void* user_struct, struct stream_state* stream);
int close_pixel_stream_hook(void* user_struct, struct stream_state* stream);
int next_row_hook(void* user_struct, struct stream_state* stream);
int next_frame_hook(void* user_struct, struct stream_state* stream);


//======================================================================
// helper functions:
//======================================================================

/*
 MergeStrings(num_args, str1, ...):
 Function that takes the count of strings to be merged and the variable
 number of strings as input, allocates memory of sufficient size and
 merges all input strings into that allocated memory.
 The return value is the pointer of the new created merged string.  If
 the first argument is just a NULL pointer, then NULL is returned.
 */
static char* MergeStrings(int num_args, char* str1, ...);

/*
 Simple error function.
 */
void bailout(char *msg);



//======================================================================
// function definitions
//======================================================================


// cameraSearchBlob():
// Take a picture and searches there for a blob with the given color.
// If no blob is found, the size is set to sero.
// Mem: This function automatically deletes the the image data.
TBlobSearch cameraSearchBlob(const char color[3])
{
    TJImage img;
    img = capturePhoto();
    return imageSearchBlob(color, &img);
} // cameraSearchBlob()


// imageSearchBlob():
// Search in an image for the maximum large blob with the given color.
// If no blob is found, the size is set to sero.
TBlobSearch imageSearchBlob(const char color[3], TJImage *pimg)
{
    TBlobSearch blob_res; // return value
    TQuickBlob dblob; // interface with quickblob

    dblob.pimg = pimg;
    dblob.ref[0] = color[0];
    dblob.ref[1] = color[1];
    dblob.ref[2] = color[2];

    extract_image((void*)&dblob); // quickblob: search blobs

    //printf("\nMax blob found:\n");
    //printf("c=<%.2f,%.2f>, size=%i, color=%i", dblob.blob_max.center_x, dblob.blob_max.center_y, dblob.blob_max.size, dblob.blob_max.color);
    //printf(", bb=<%i,%i>-<%i,%i>", dblob.blob_max.bb_x1, dblob.blob_max.bb_y1, dblob.blob_max.bb_x2, dblob.blob_max.bb_y2);
    //printf(", alignment=<%f,%f>", dblob.halign, dblob.valign);
    //printf("\n");

    // copy relevant data into return value
    blob_res.blob = dblob.blob_max;
    blob_res.size = dblob.blob_max.size;
    if (dblob.blob_max.size > 0) {
        // blob found --> calculate its alignment to center;
        blob_res.halign = -1.0 + 2.0* ((double)(dblob.blob_max.center_x) / (dblob.pimg)->w);
        blob_res.valign = -1.0 + 2.0* ((double)(dblob.blob_max.center_y) / (dblob.pimg)->h);
    }   
    blob_res.pimg = pimg;

    return blob_res;
} // imageSearchBlob()


// read_JPEG_image():
// Function to read jpeg image data (using libjpeg)
// Mem: The data buffer of the returned image gets overwritten on each call.
TJImage read_JPEG_image (FILE *file) {
    struct jpeg_decompress_struct info; //for our jpeg info
    struct jpeg_error_mgr err; //the error handler
    static unsigned char *img_data = NULL;

    TJImage img;               // image data structure

    info.err = jpeg_std_error( &err );     
    jpeg_create_decompress( &info ); //fills info structure
    jpeg_stdio_src( &info, file );    
    jpeg_read_header( &info, TRUE );
    jpeg_start_decompress( &info );

    img.w = info.output_width;
    img.h = info.output_height;
    img.numChannels = info.num_components; // 3 = RGB, 4 = RGBA
    unsigned long dataSize = img.w * img.h * img.numChannels;
    // adjust size of image data buffer
    if (img_data == NULL) {
        img_data = (unsigned char *) malloc( dataSize );
    }
    else {
        img_data = (unsigned char *) realloc( img_data, dataSize );
    }
    img.data = img_data;

    // read RGB(A) scanlines one at a time into jdata[]
    unsigned char* rowptr;
    while ( info.output_scanline < img.h )
    {
        rowptr = img.data + info.output_scanline * img.w * img.numChannels;
        jpeg_read_scanlines( &info, &rowptr, 1 );
    }

    jpeg_finish_decompress( &info );    

    // code block to illustrate image traversal 
    /*
    for ( int x = 0 ; x < img.w ; x++ )
        for ( int y = 0 ; y < img.h ; y++ )
            for ( int c = 0 ; c < img.numChannels ; c++ )
                dat = data[ y * img.w * img.numChannels + x * img.numChannels + c ];
    */
    //fprintf(stdout, "Image data: width=%d, height=%d, #channels=%d\n", img.w, img.h, img.numChannels );
    return img;
} // read_JPEG_image()


// readJpegImageFromFile():
// Function to read jpeg image data (using libjpeg)
// Mem: The data buffer of the returned image gets overwritten on each call.
TJImage readJpegImageFromFile (const char *fname)
{
    FILE *file;
    TJImage img;

    file = fopen( fname, "rb" );
    if ( file == NULL )
    {
        exit(EXIT_FAILURE);
    }
    img = read_JPEG_image (file);
    fclose( file );

    //fprintf(stdout, "Image data: width=%d, height=%d, #channels=%d\n", img.w, img.h, img.numChannels );

    return img;
} // readJpegImageFromFile()


// capturePhotoToFile():
// Take a picture via RasperiPI camera and save it as a .jpg file.
int capturePhotoToFile(const char *fname)
{
    FILE *fp;
    char *cmd;
    cmd = MergeStrings( 4, CAMERA_CMD, " -o ", fname, " " ); 
    fp = popen(cmd, "r");

    if (fp == NULL) bailout("capturePhotoToFile() failed!");

    pclose(fp);
    free(cmd); // cleanup
    return 0; 
} // capturePhotoToFile()


// capturePhoto():
// Take a picture via RasperiPI camera and return the raw image data
// Mem: The meory for the image data needs to be explicitly freed.
TJImage capturePhoto()
{
    FILE *fp;
    TJImage img;
    char *cmd;
    cmd = MergeStrings( 2, CAMERA_CMD, " -o - " ); 
    fp = popen(cmd, "r");

    if (fp == NULL) bailout("capturePhoto() failed!");

    img = read_JPEG_image (fp);

    pclose(fp);
    free(cmd); // cleanup
    return img;
} // capturePhoto()


// Function to save a loaded image as JPEG file
// quality: integer 0..100
void writeImageAsJPEG(TJImage *pimg, const char *fname, int quality)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    // JSAMPROW ... "unsigned char *"
    JSAMPROW row_pointer;	/* pointer to JSAMPLE row[s] */
    int row_stride;
    cinfo.err = jpeg_std_error(&jerr);

    outfile = fopen(fname, "wb");
    if (outfile == NULL) bailout("writeImageAsJPEG: error opening file");

    jpeg_create_compress ( &cinfo );
    jpeg_stdio_dest ( &cinfo, outfile );
 
    cinfo.image_width      = pimg->w;
    cinfo.image_height     = pimg->h;
    cinfo.input_components = pimg->numChannels;
    cinfo.in_color_space   = JCS_RGB;    

    jpeg_set_defaults(&cinfo);
    /*set the quality [0..100]  */
    jpeg_set_quality ( &cinfo, quality, TRUE);
    jpeg_start_compress ( &cinfo, TRUE);

    row_stride = pimg->w * pimg->numChannels;

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW) & pimg->data[ cinfo.next_scanline * row_stride ];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
}


// Function to mark a loaded image with a blob and save it as JPEG file
void writeImageWithBlobAsJPEG(TBlobSearch blobsearch, const char *fname, int quality)
{
    TJImage *pimg = blobsearch.pimg;
    int x,y,c;
    const double image_scale = 0.7;  // factor, by which to scale image colors to allow for contrast with blob
    int x1,x2,y1,y2;

    // sweep over all pixels to make them brighter (to have better contrast for marking blob in black)
    for ( x = 0 ; x < pimg->w ; x++ ) {
        for ( y = 0 ; y < pimg->h ; y++ ) {
            for ( c = 0 ; c < pimg->numChannels ; c++ ) {
                JImageDATA(pimg,x,y,c) = (1-image_scale)*255 + image_scale * JImageDATA(pimg,x,y,c);
            }
        }
    }
    // write rectancle with blob information
    x1 = min( blobsearch.blob.bb_x1, blobsearch.blob.bb_x2 );
    x2 = max( blobsearch.blob.bb_x1, blobsearch.blob.bb_x2 );
    y1 = min( blobsearch.blob.bb_y1, blobsearch.blob.bb_y2 );
    y2 = max( blobsearch.blob.bb_y1, blobsearch.blob.bb_y2 );
    for ( x = x1 ; x <= x2 ; x++ ) {
        JImageDATA(pimg,x,y1,0) = 0; // R, 1st hor line
        JImageDATA(pimg,x,y1,1) = 0; // G, 1st hor line
        JImageDATA(pimg,x,y1,2) = 0; // B, 1st hor line
        JImageDATA(pimg,x,y2,0) = 0; // R, 2nd hor line
        JImageDATA(pimg,x,y2,1) = 0; // G, 2nd hor line
        JImageDATA(pimg,x,y2,2) = 0; // B, 2nd hor line
    }
    for ( y = y1 ; y <= y2 ; y++ ) {
        JImageDATA(pimg,x1,y,0) = 0; // R, 1st vert line
        JImageDATA(pimg,x1,y,1) = 0; // G, 1st vert line
        JImageDATA(pimg,x1,y,2) = 0; // B, 1st vert line
        JImageDATA(pimg,x2,y,0) = 0; // R, 2nd vert line
        JImageDATA(pimg,x2,y,1) = 0; // G, 2nd vert line
        JImageDATA(pimg,x2,y,2) = 0; // B, 2nd vert line
    }
    writeImageAsJPEG(pimg, fname, quality);  // write marked image to file
}


// Function to save a loaded image as a CSV (comma separated value) 
// text file.  This function might be useful to analyse the light situation
// for the camera.  Note that this may produce much large files than the
// original image.
void writeImageAsCSV(TJImage *pimg, const char *fname)
{
    FILE *fp;
    fp = fopen (fname, "wb");
    int x,y,c;
    unsigned char dat;

    fprintf(fp,"X, Y, Red, Green, Blue");
    if (pimg->numChannels == 4)
        { fprintf(fp,", Alpha"); }
    fprintf(fp,"\n");

    for ( x = 0 ; x < pimg->w ; x++ ) {
        for ( y = 0 ; y < pimg->h ; y++ ) {
            fprintf(fp,"%u, %u", x, y);
            for ( c = 0 ; c < pimg->numChannels ; c++ ) {
                dat = pimg->data[ y * pimg->w * pimg->numChannels + x * pimg->numChannels + c ];
                fprintf(fp,", %u", dat);
            }
            fprintf(fp,"\n");
        }
    }
    fclose (fp);
}





//======================================================================
// some local hook functions to interface with QuickBlob library:
//======================================================================


void log_blob_hook(void* user_struct, struct blob* b)
// center_x, center_y and size are the cumulative stats for a blob
{
    TQuickBlob *opt = user_struct;

    if (b->color == 255 && b->size > (opt->blob_max).size)
        { opt->blob_max = *b; };
    //printf("seg: y=%i,x=%i...%i, ", b->y, b->x1, b->x2);
    //printf("c=<%.2f,%.2f>, size=%i, color=%i", b->center_x, b->center_y, b->size, b->color);
    //printf(", bb=<%i,%i>-<%i,%i>", b->bb_x1, b->bb_y1, b->bb_x2, b->bb_y2);
    //printf("\n");
}

int init_pixel_stream_hook(void* user_struct, struct stream_state* stream)
// get the image ready for streaming
// set the width and height
{
    TQuickBlob *opt = user_struct;
    stream->w = (opt->pimg)->w;
    stream->h = (opt->pimg)->h-1;
    opt->frame = -1;
    (opt->blob_max).size = 0;
    // calc ref. color relative to red component
    opt->ref_rel[0] = 1.0;
    opt->ref_rel[1] = opt->ref[1] / max(opt->ref[0], 1); //avoid div.by zero
    opt->ref_rel[2] = opt->ref[2] / max(opt->ref[0], 1); //avoid div.by zero
    return 0;
} // log_blob_hook()

int close_pixel_stream_hook(void* user_struct, struct stream_state* stream)
// free up anything you allocated in init_pixel_stream_hook
// return status (0 for success)
{
    //TQuickBlob *opt = user_struct;
    return 0;
} // close_pixel_stream_hook()

int next_row_hook(void* user_struct, struct stream_state* stream)
// load the (grayscale) row at stream->y into the (8 bit) stream->row array
// return status (0 for success)
{
    TQuickBlob *opt = user_struct;
    int x;
    int pixel[3]; // rgb components of pixel
    //const double flimit = 1.2;
    for (x=0; x < stream->w; x++) {
        // copy current row and convert into B/W color
        pixel[0] = JImageDATA ((opt->pimg),x,stream->y,0);
        pixel[1] = JImageDATA ((opt->pimg),x,stream->y,1);
        pixel[2] = JImageDATA ((opt->pimg),x,stream->y,2);
        if (
             // The following test for generic colors does not work, as color variation
             // due to light conditions is too much:
             //BLOB_MATCH(opt->ref[0], pixel[0]) &&
             //BLOB_MATCH(opt->ref[1], pixel[1]) &&
             //BLOB_MATCH(opt->ref[2], pixel[2])

             // Alternative: a specific test that seems to work for color RED indoors:
             pixel[0] >= 100 && pixel[1] <= 80 && pixel[2] <= 80
            )
            stream->row[x] = 255;
        else
            stream->row[x] = 0;
    }
    return 0;
} // next_row_hook()

int next_frame_hook(void* user_struct, struct stream_state* stream)
// basically a no-op in the library, but useful for applications
// return status (0 for success, otherwise breaks the video loop)
{
    TQuickBlob *opt = user_struct;
    opt->frame++;
    return opt->frame;
}


/*
 MergeStrings(num_args, str1, ...):
 Function that takes the count of strings to be merged and the variable
 number of strings as input, allocates memory of sufficient size and
 merges all input strings into that allocated memory.
 The return value is the pointer of the new created merged string.  If
 the first argument is just a NULL pointer, then NULL is returned.
 Updated: 01.10.2023
 */
static char* MergeStrings(int num_args, char* str1, ...)
{
   va_list ap;
   char *p, *pins;
   int len = 0;
   int i;
   char *smerged = NULL;

   len = strlen(str1) ;
   va_start(ap, str1) ;
   for(i = num_args - 1; i > 0; i--)
   {
      p = va_arg(ap, char*) ;
      len = len + strlen(p) ;
   }
   va_end(ap) ;

   if (len > 0)
   {
      len = len + 1 ;  // allocate extra space for string termination
      smerged = malloc (len) ;

      if (smerged==NULL) return NULL ;  // out-of-memory?
      smerged[0] = '\0';  // init as empty string

      pins = smerged;  // points to first empty space in array
      strcpy(pins, str1) ;
      pins = pins + strlen(str1) ;
      va_start(ap, str1) ;
      for(i = num_args -1; i > 0; i--)
      {
         p = va_arg(ap, char*) ;
         strcpy(pins, p) ;
         pins = pins + strlen(p) ;
      }
      va_end(ap) ;
      *pins = '\0' ;  // terminate the merged string
   }

   return smerged ;  // return allocated memory (needs explicit free() later)
}


/*
 Simple error function.
 */
void bailout(char *msg)
{
   fprintf(stderr,"\nFatal Error: %s\n\n",msg);
   exit(EXIT_FAILURE);
}

