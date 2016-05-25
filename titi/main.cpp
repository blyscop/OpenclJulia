//
//  main.cpp
//  Julia
//
//  Created by Yanick Servant on 25/05/2016.
//  Copyright © 2016 Yanick Servant. All rights reserved.
//


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif
#include "ocl_macros.h"

#include "../../common/cpu_bitmap.h"

//Common defines
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

//Custom defines
#define N 10
#define DIM 1000

const char *julia =
"typedef struct cuComplex {                 \n"
"float   r;                 \n"
"float   i;                 \n"
"} cuComplex;                 \n"
"                 \n"
"cuComplex createComplex(float _r, float _i) {                 \n"
"    struct cuComplex tmp;                 \n"
"    tmp.r = _r;                 \n"
"    tmp.i = _i;                 \n"
"                 \n"
"    return tmp;                 \n"
"}                 \n"
"                 \n"
"float magnitude2(cuComplex z) {                 \n"
"    return z.r * z.r + z.i * z.i;                 \n"
"}                 \n"
"                 \n"
"cuComplex multiply(cuComplex a, cuComplex b) {                 \n"
"    return createComplex(a.r * b.r - a.i * b.i, a.i * b.r + a.r * b.i);                 \n"
"}                 \n"
"                 \n"
"cuComplex add(cuComplex a, cuComplex b) {                 \n"
"    return createComplex(a.r + b.r, a.i + b.i);                 \n"
"}                 \n"
"                 \n"
"int julia(int x, int y) {                 \n"
"    const float scale = 1.5;                 \n"
"    float jx = scale * (float)(1000 / 2 - x) / (1000 / 2);                 \n"
"    float jy = scale * (float)(1000 / 2 - y) / (1000 / 2);                 \n"
"                 \n"
"    cuComplex c = createComplex(-0.8, 0.156);                 \n"
"    cuComplex a = createComplex(jx, jy);                 \n"
"                 \n"
"    int i = 0;                 \n"
"    for (i = 0; i<200; i++) {                 \n"
"        a = add(multiply(a, a), c);                 \n"
"        if (magnitude2(a) > 1000)                 \n"
"            return 0;                 \n"
"    }                 \n"
"                 \n"
"    return 1;                 \n"
"}                 \n"
"__kernel                                   \n"
"void kernelcpu(__global char *ptr){                \n"
"                \n"
"    int x = get_global_id(0);                \n"
"    int y = get_global_id(1);                \n"
"                \n"
"    int offset =x + y * 1000;                \n"
"    int jVal = julia(x , y);                \n"
"                \n"
"    ptr[offset * 4+0] = jVal * 255;                \n"
"    ptr[offset * 4+1] = 0;                \n"
"    ptr[offset * 4+2] = 0;                \n"
"    ptr[offset * 4+3] = 255 ;                \n"
"}  \n";


int main(void) {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();
    
    //kernelcpu(ptr);
    
    cl_int clStatus; //Keeps track of the error values returned.
    
    // Get platform and device information
    cl_platform_id * platforms = NULL;
    
    // Set up the Platform. Take a look at the MACROs used in this file.
    // These are defined in common/ocl_macros.h
    OCL_CREATE_PLATFORMS(platforms);
    
    // Get the devices list and choose the type of device you want to run on
    cl_device_id *device_list = NULL;
    OCL_CREATE_DEVICE(platforms[0], DEVICE_TYPE, device_list);
    
    // Create OpenCL context for devices in device_list
    cl_context context;
    cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[0],
        0
    };
    // An OpenCL context can be associated to multiple devices, either CPU or GPU
    // based on the value of DEVICE_TYPE defined above.
    context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed...");
    
    // Create a command queue for the first device in device_list
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed...");
    
    // Create memory buffers on the device for each vector
    cl_mem ptr_clem = clCreateBuffer(context, CL_MEM_READ_WRITE, 1000 * 1000 * 4 * sizeof(char), NULL, &clStatus);
    
    // Copy the Buffer A and B to the device. We do a blocking write to the device buffer.
    clStatus = clEnqueueWriteBuffer(command_queue, ptr_clem, CL_TRUE, 0, 1000 * 1000 * 4 * sizeof(char), ptr, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed...");
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&julia, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed...");
    
    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if (clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "kernelcpu", &clStatus);
    
    // Set the arguments of the kernel. Take a look at the kernel definition in sum_event
    // variable. First parameter is a constant and the other three are buffers.
    clStatus |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &ptr_clem);
    LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed...");
    
    // Execute the OpenCL kernel on the list
    size_t global_size[2] = {1000, 1000};
    size_t local_size = 1;
    cl_event sum_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &sum_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed...");
    
    // Read the memory buffer C_clmem on the device to the host allocated buffer C
    // This task is invoked only after the completion of the event sum_event
    clStatus = clEnqueueReadBuffer(command_queue, ptr_clem, CL_TRUE, 0, sizeof(char) * 1000 * 1000 * 4, ptr, 1, &sum_event, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed...");
    
    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);
    
    // Display the result to the screen
    
    // Finally release all OpenCL objects and release the host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(ptr_clem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(platforms);
    free(device_list);
    
    //return 0;
    
    
    bitmap.display_and_exit();
    
}