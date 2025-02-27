#pragma once

#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define RGBAU8_IMAGE_SIZE(width, height) ((width) * (height) * 4)

typedef enum  {
    SUCCESS = 0,
    FAILED = 1
}Result;

typedef enum{
    RGB = 1,
    GRAY = 2,
}INPUT_TYPE;



typedef struct  {
    float x;
    float y;
    float w;
    float h;
    float score;
    int classIndex;
    int index; // index of output buffer
} BBox;

typedef struct tag_ImgMat
{
    /* data */
    unsigned char* data = NULL;
    // bool relayout=false;
}ImgMat;